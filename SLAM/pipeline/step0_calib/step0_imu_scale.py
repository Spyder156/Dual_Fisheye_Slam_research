#!/usr/bin/env python
"""Recover METRIC scale for a Step 0 reconstruction using the IMU.

WHY: SfM scale is arbitrary, so the rig baseline comes out in meaningless units.
Hilti has ground truth we can align to, but an Insta360 will not -- the whole
point of Step 0 is per-unit calibration with no external reference. The IMU is
the only metric sensor on the rig.

HOW (standard visual-inertial alignment, as in VINS-Mono / ORB-SLAM3 init):
Between consecutive keyframes i,i+1 preintegrate the IMU to get
    dP_i (double-integrated accel), dV_i (integrated accel), dt_i
all in the BODY frame, bias-free terms separated so bias enters linearly.

With s = scale (SfM -> metres), g = gravity in the SfM world frame, ba = accel
bias, and v_i = body velocity at keyframe i, the position relation is

    s*(p_{i+1} - p_i) = R_i*(dP_i - Jp_ba*ba) + v_i*dt_i + 0.5*g*dt_i^2

and the velocity relation

    0 = R_i*(dV_i - Jv_ba*ba) + (v_{i+1} - v_i)*(-1) + g*dt_i     [rearranged]

Both are LINEAR in [s, g(3), ba(3), v_0..v_n(3 each)], so we stack them and
solve one least-squares problem. |g| = 9.81 is what makes the solution metric:
without it scale and gravity trade off exactly.

Usage:
  step0_imu_scale.py --sparse <model_dir> --dataset <ds_dir> [--truth-scale 2.4422]
"""
import argparse
import struct
from pathlib import Path

import numpy as np

G = 9.81007


def read_images_bin(p):
    out = {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<i", f.read(4))
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            struct.unpack("<i", f.read(4))
            nm = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                nm += c
            k = struct.unpack("<Q", f.read(8))[0]
            f.read(k * 24)
            out[nm.decode()] = (q, t)
    return out


def q2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def preintegrate(t_imu, acc, gyr, t0, t1):
    """Body-frame preintegration between t0 and t1 (bias-free part) plus the
    Jacobians wrt accel bias, which enter the solve linearly."""
    m = (t_imu >= t0) & (t_imu < t1)
    if m.sum() < 2:
        return None
    ts, a, w = t_imu[m], acc[m], gyr[m]
    R = np.eye(3)
    dV = np.zeros(3); dP = np.zeros(3)
    Jv = np.zeros((3, 3)); Jp = np.zeros((3, 3))
    for k in range(len(ts) - 1):
        dt = ts[k+1] - ts[k]
        if dt <= 0 or dt > 0.1:
            continue
        dP += dV * dt + 0.5 * (R @ a[k]) * dt * dt
        Jp += Jv * dt - 0.5 * R * dt * dt
        dV += (R @ a[k]) * dt
        Jv += -R * dt
        th = w[k] * dt
        n = np.linalg.norm(th)
        if n > 1e-12:
            kx = th / n
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            R = R @ (np.eye(3) + np.sin(n) * K + (1 - np.cos(n)) * K @ K)
    return dP, dV, Jp, Jv, ts[-1] - ts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--truth-scale", type=float, default=None)
    ap.add_argument("--truth-baseline-cm", type=float, default=None)
    ap.add_argument("--baseline-sfm", type=float, default=None)
    ap.add_argument("--max-kf", type=int, default=200)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="write align.json (scale + gravity_dir) for make_imu_factors")
    ap.add_argument("--imucam", type=Path, default=None,
                    help="kalibr_imucam_chain.yaml -- REQUIRED: preintegration is in "
                         "the IMU body frame, so camera rotations must be mapped "
                         "through R_cam->imu or gravity leaks into the accel bias.")
    args = ap.parse_args()

    # ---- camera poses (cam0) at known times ----
    imgs = read_images_bin(args.sparse / "images.bin")
    fr = np.genfromtxt(args.dataset / "frames.csv", delimiter=",", names=True)
    tmap = dict(zip(np.atleast_1d(fr["frame"]).astype(int), np.atleast_1d(fr["t"])))
    rows = []
    for n, (q, t) in imgs.items():
        # accept both layouts: Step 0 uses "cam0/000001.jpg", the triangulated
        # model from a SLAM trajectory uses flat "000001.jpg"
        if "/" in n and not n.startswith("cam0/"):
            continue
        k = int(n.split("/")[-1].split(".")[0])
        if k in tmap:
            R = q2R(q)
            rows.append((tmap[k], -R.T @ t, R.T))       # t, camera centre, R_cam->world
    # camera -> IMU frame. Without this the preintegrated body-frame terms are
    # rotated by the wrong basis and the solve puts gravity into the bias.
    R_CtoI = np.eye(3); p_CinI = np.zeros(3)
    if args.imucam:
        import re
        txt = args.imucam.read_text()
        blk = re.split(r"\ncam\d+:", txt)[1]
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                             blk.split("cam_overlaps")[0])]
        Tm = np.array(nums).reshape(4, 4)
        R_CtoI, p_CinI = Tm[:3, :3], Tm[:3, 3]
        print(f"cam->imu loaded: |p_CinI| = {np.linalg.norm(p_CinI)*100:.2f} cm")
    # R_CtoI maps cam->imu, so R_{W<-I} = R_{W<-C} @ R_CtoI^T, and the IMU
    # position is the camera centre minus the lever arm in world coords.
    conv = []
    for (t, C, Rcw) in rows:
        R_WI = Rcw @ R_CtoI.T
        conv.append((t, C - R_WI @ p_CinI, R_WI))
    rows = conv
    rows.sort(key=lambda r: r[0])
    rows = rows[:args.max_kf]
    T = np.array([r[0] for r in rows])
    P = np.array([r[1] for r in rows])
    Rw = np.array([r[2] for r in rows])
    print(f"{len(rows)} keyframes, {T[-1]-T[0]:.1f} s span")

    # ---- IMU ----
    d = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti = d["t"]; gyr = np.stack([d["gx"], d["gy"], d["gz"]], 1)
    acc = np.stack([d["ax"], d["ay"], d["az"]], 1)

    n = len(T) - 1
    # unknowns: s(1) g(3) ba(3) v_0..v_n (3*(n+1))
    nx = 1 + 3 + 3 + 3 * (n + 1)
    A = []; b = []
    used = 0
    for i in range(n):
        pi = preintegrate(ti, acc, gyr, T[i], T[i+1])
        if pi is None:
            continue
        dP, dV, Jp, Jv, dt = pi
        if dt <= 0:
            continue
        used += 1
        Ri = Rw[i]
        # position:  s*(p1-p0) - v_i*dt - 0.5*g*dt^2 + Ri*Jp*ba = Ri*dP
        r = np.zeros((3, nx))
        r[:, 0:1] = (P[i+1] - P[i]).reshape(3, 1)
        r[:, 1:4] = -0.5 * np.eye(3) * dt * dt
        r[:, 4:7] = Ri @ Jp
        r[:, 7+3*i:10+3*i] = -np.eye(3) * dt
        A.append(r); b.append(Ri @ dP)
        # velocity:  v_{i+1} - v_i - g*dt + Ri*Jv*ba = Ri*dV
        r = np.zeros((3, nx))
        r[:, 1:4] = -np.eye(3) * dt
        r[:, 4:7] = Ri @ Jv
        r[:, 7+3*i:10+3*i] = -np.eye(3)
        r[:, 10+3*i:13+3*i] = np.eye(3)
        A.append(r); b.append(Ri @ dV)
    A = np.vstack(A); b = np.concatenate(b)
    print(f"{used}/{n} intervals had IMU coverage; system {A.shape}")

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    s, g, ba = x[0], x[1:4], x[4:7]
    print(f"\nraw solve:  scale {s:.4f}   |g| {np.linalg.norm(g):.3f} (expect {G:.3f})   "
          f"ba {np.round(ba,4)}")

    # enforce |g| = 9.81: rescale so gravity has the right magnitude, which is
    # what actually pins the metric scale
    if np.linalg.norm(g) > 1e-6:
        k = G / np.linalg.norm(g)
        print(f"gravity-normalised scale: {s*k:.4f}   (correction x{k:.4f})")
        s_final = s * k
    else:
        s_final = s

    if args.out_json:
        import json
        # NEGATED: our solve returns the UP direction (accelerometer specific
        # force at rest points up). make_imu_factors wants gravity pointing DOWN
        # -- feeding it un-negated put 2g into the accel bias (|b_a| = 18.9).
        gdir = (-g / np.linalg.norm(g)).tolist()
        args.out_json.write_text(json.dumps(
            {"scale": float(s_final), "gravity_dir": gdir,
             "gravity_norm": float(np.linalg.norm(g)),
             "accel_bias": ba.tolist()}, indent=1))
        # velocities sidecar: make_imu_factors reads <stem>_velocities.csv and
        # WITHOUT it every velocity inits to zero, so the accel bias absorbs the
        # mismatch (seen: |b_a| ~ 19, i.e. two gravities).
        v = x[7:].reshape(-1, 3) * (k if np.linalg.norm(g) > 1e-6 else 1.0)
        vt = T[:len(v)]
        vcsv = args.out_json.with_name(args.out_json.stem + "_velocities.csv")
        np.savetxt(vcsv, np.column_stack([vt, v]), header="t,vx,vy,vz",
                   delimiter=",", comments="")
        print(f"wrote {args.out_json}  gravity_dir={np.round(gdir,4)}")
        print(f"wrote {vcsv}  {len(v)} velocities, median speed "
              f"{np.median(np.linalg.norm(v,axis=1)):.2f} m/s")

    if args.truth_scale:
        print(f"\n  truth scale (from GT Sim3): {args.truth_scale:.4f}")
        print(f"  error: {100*(s_final-args.truth_scale)/args.truth_scale:+.1f} %")
    if args.baseline_sfm:
        print(f"\n  rig baseline: {100*s_final*args.baseline_sfm:.2f} cm", end="")
        if args.truth_baseline_cm:
            print(f"   (truth {args.truth_baseline_cm:.2f} cm, "
                  f"error {100*s_final*args.baseline_sfm-args.truth_baseline_cm:+.2f} cm)")
        else:
            print()


if __name__ == "__main__":
    main()
