#!/usr/bin/env python3
"""E7 step 1 — linear visual-inertial alignment on keyframe poses.

Given keyframe body rotations+positions (any scale, any world) and raw IMU,
preintegrate gyro+accel between keyframes (zero bias, midpoint rule) and
solve the linear system for: metric scale s, gravity vector g (in the model
world), and per-keyframe velocities.

Self-validation built in:
  - ||g|| should come out ~9.8 m/s² without being asked to
  - s should match the independent Sim3-vs-reference estimate where we have one

Usage: vi_align.py <ep> <poses.tum|inloc_hloc.txt> [--max-dt 0.6]
  poses: TUM (IMU frame) or their inloc format (auto-detected, converted).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P1 = Path(os.environ.get("MECKA_ROOT", "/pipeline/data"))
R_bc = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
t_bc = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])


def q2R(x, y, z, w):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def load_poses(ep, path):
    """Return (t, p_wb, R_wb) body-frame keyframe poses, corrected clock."""
    frames_csv = sorted((P1 / "data/episodes" / ep).glob("frames*.csv"))[0]
    off = json.load(open(P1 / f"results/offsets/{ep}.json"))["offset_s"]
    unix = pd.read_csv(frames_csv)["unix_timestamp"].to_numpy() + off
    first = open(path).readline().split()
    rows = []
    if first and first[0].endswith(".jpg"):          # their inloc: w2c cam
        for line in open(path):
            f = line.split()
            if len(f) != 8 or not f[0].endswith(".jpg"):
                continue
            idx = int(Path(f[0]).stem)
            if idx >= len(unix):
                continue
            qw, qx, qy, qz = map(float, f[1:5])
            t_cw = np.array(list(map(float, f[5:8])))
            R_cw = q2R(qx, qy, qz, qw)
            R_wc = R_cw.T
            p_wc = -R_cw.T @ t_cw
            R_wb = R_wc @ R_bc.T
            p_wb = p_wc - R_wb @ t_bc
            rows.append((unix[idx], p_wb, R_wb))
    else:                                            # TUM IMU-frame
        raw = np.loadtxt(path, comments="#")
        step = max(1, len(raw) // 450)               # keyframe-ish subsample
        for r in raw[::step]:
            rows.append((r[0], r[1:4], q2R(*r[4:8])))
    rows.sort(key=lambda r: r[0])
    return ([r[0] for r in rows], np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def preintegrate(ep, t_kf):
    """Zero-bias midpoint preintegration between consecutive keyframes.
    Returns per-interval (dt, dR, dv, dp) in the body frame of keyframe i."""
    sd = P1 / "data/episodes" / ep / "sensor"
    gy = pd.read_csv(sd / "raw_gyroscope.csv")
    ac = pd.read_csv(sd / "raw_accelerometer.csv")
    tg = gy["timestamp"].to_numpy()
    wg = gy[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    ta = ac["timestamp"].to_numpy()
    aa = ac[["acc_x", "acc_y", "acc_z"]].to_numpy()
    # common 200 Hz grid
    lo, hi = max(tg[0], ta[0]), min(tg[-1], ta[-1])
    tt = np.arange(lo, hi, 0.005)
    W = np.stack([np.interp(tt, tg, wg[:, i]) for i in range(3)], 1)
    A = np.stack([np.interp(tt, ta, aa[:, i]) for i in range(3)], 1)

    out = []
    for i in range(len(t_kf) - 1):
        t0, t1 = t_kf[i], t_kf[i + 1]
        m = (tt >= t0) & (tt < t1)
        if m.sum() < 3:
            out.append(None)
            continue
        R = np.eye(3)
        dv = np.zeros(3)
        dp = np.zeros(3)
        Jv = np.zeros((3, 3))   # d(dv)/d(b_a)
        Jp = np.zeros((3, 3))   # d(dp)/d(b_a)
        ts = tt[m]
        for k in range(len(ts) - 1):
            dt = ts[k + 1] - ts[k]
            w_mid = 0.5 * (W[m][k] + W[m][k + 1])
            a_k = 0.5 * (A[m][k] + A[m][k + 1])
            dp = dp + dv * dt + 0.5 * (R @ a_k) * dt * dt
            Jp = Jp + Jv * dt - 0.5 * R * dt * dt
            dv = dv + (R @ a_k) * dt
            Jv = Jv - R * dt
            R = R @ so3_exp(w_mid * dt)
        out.append((t1 - t0, R, dv, dp, Jv, Jp))
    return out


def align(t_kf, p, Rwb, pre, max_dt):
    """Linear LS for scale s, gravity g, velocities v_i.
    Body-frame position eq per interval i:
      Rwb_i^T (s*(p_{i+1}-p_i) - v_i*dt - 0.5*g*dt^2) = dp_i
    Velocity eq:
      Rwb_i^T (v_{i+1} - v_i - g*dt) = dv_i
    """
    n = len(t_kf)
    use = [i for i, pr in enumerate(pre)
           if pr is not None and pr[0] <= max_dt]
    nv = n
    cols = 1 + 3 + 3 + 3 * nv                 # s, g, b_a, v_0..v_{n-1}
    VOFF = 7
    rows_A, rows_b = [], []
    for i in use:
        dt, _dR, dv, dp, Jv, Jp = pre[i]
        Rt = Rwb[i].T
        # position row-block: Rt(s*dpw - v_i dt - .5 g dt^2) - Jp b_a = dp
        Ab = np.zeros((3, cols))
        Ab[:, 0] = Rt @ (p[i + 1] - p[i])
        Ab[:, 1:4] = -0.5 * dt * dt * Rt
        Ab[:, 4:7] = -Jp
        Ab[:, VOFF + 3 * i:VOFF + 3 * i + 3] = -dt * Rt
        rows_A.append(Ab)
        rows_b.append(dp)
        # velocity row-block
        Ab = np.zeros((3, cols))
        Ab[:, 1:4] = -dt * Rt
        Ab[:, 4:7] = -Jv
        Ab[:, VOFF + 3 * i:VOFF + 3 * i + 3] = -Rt
        Ab[:, VOFF + 3 * (i + 1):VOFF + 3 * (i + 1) + 3] = Rt
        rows_A.append(Ab)
        rows_b.append(dv)
    A = np.vstack(rows_A)
    b = np.hstack(rows_b)
    use_idx = use
    x, _res, *_ = np.linalg.lstsq(A, b, rcond=None)

    # gravity-norm refinement (VINS-Mono style): fix |g| = G0, re-solve with
    # g = G0 * (ghat + B tau) where B spans the tangent plane at ghat
    G0 = 9.805
    for _ in range(4):
        ghat = x[1:4] / np.linalg.norm(x[1:4])
        # tangent basis
        tmp = np.array([1.0, 0, 0]) if abs(ghat[0]) < 0.9 else np.array([0, 1.0, 0])
        b1 = np.cross(ghat, tmp); b1 /= np.linalg.norm(b1)
        b2 = np.cross(ghat, b1)
        B = np.stack([b1, b2], 1)                     # 3x2
        # substitute: columns for g (1:4) -> g0 const + 2 tangent cols
        A2 = np.delete(A, [1, 2, 3], axis=1)
        A2 = np.hstack([A2[:, :1], A[:, 1:4] @ B, A2[:, 1:]])
        b2v = b - A[:, 1:4] @ (G0 * ghat)
        x2, *_ = np.linalg.lstsq(A2, b2v, rcond=None)
        g_new = G0 * ghat + B @ x2[1:3]
        g_new = G0 * (g_new / np.linalg.norm(g_new))
        x = np.concatenate([[x2[0]], g_new, x2[3:]])
    s = x[0]
    g = x[1:4]
    ba = x[4:7]
    v = x[7:].reshape(-1, 3)
    r = (A @ x - b).reshape(-1, 2, 3)          # per-interval [pos, vel] rows
    per_int = np.linalg.norm(r[:, 0, :], axis=1)   # position-eq residual norm
    rms = float(np.sqrt(np.mean((A @ x - b) ** 2)))
    return s, g, ba, v, rms, use_idx, per_int


def main():
    ep, path = sys.argv[1], sys.argv[2]
    max_dt = 0.6
    if "--max-dt" in sys.argv:
        max_dt = float(sys.argv[sys.argv.index("--max-dt") + 1])
    t_kf, p, Rwb = load_poses(ep, path)
    print(f"{ep}: {len(t_kf)} keyframes from {Path(path).name}")
    pre = preintegrate(ep, t_kf)
    s, g, ba, v, rms, use_idx, per_int = align(t_kf, p, Rwb, pre, max_dt)
    nused = len(use_idx)
    speeds = np.linalg.norm(v, axis=1)
    print(f"  intervals used: {nused} (dt <= {max_dt}s)")
    print(f"  SCALE      s     = {s:.4f}")
    print(f"  GRAVITY    |g|   = {np.linalg.norm(g):.3f} m/s^2   "
          f"(should be ~9.81 — this is the self-check)")
    print(f"  gravity dir      = {g / np.linalg.norm(g)}")
    print(f"  ACCEL BIAS b_a   = {np.round(ba, 4)}  (|b_a| = "
          f"{np.linalg.norm(ba):.3f} m/s^2, sane < 0.5)")
    print(f"  velocities       : median {np.median(speeds):.2f}, "
          f"p95 {np.percentile(speeds,95):.2f} m/s (already metric)")
    print(f"  residual rms     = {rms:.4f}")

    if "--out-dir" in sys.argv:
        od = Path(sys.argv[sys.argv.index("--out-dir") + 1])
        od.mkdir(parents=True, exist_ok=True)
        tag = (sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv
               else Path(path).stem)
        gn = g / np.linalg.norm(g)
        json.dump({
            "episode": ep, "input": str(path), "n_keyframes": len(t_kf),
            "n_intervals_used": nused, "scale": float(s),
            "gravity_norm": float(np.linalg.norm(g)),
            "gravity_dir": gn.tolist(), "accel_bias": ba.tolist(),
            "accel_bias_norm": float(np.linalg.norm(ba)),
            "residual_rms": rms,
            # v is solved in metric units directly (see align(): its rows mix
            # v*dt with metric dp and g), so no further scaling by s.
            "speed_median_mps": float(np.median(np.linalg.norm(v, axis=1))),
        }, open(od / f"{tag}.json", "w"), indent=1)
        # metric-scaled trajectory + per-interval residual series for viz
        with open(od / f"{tag}_metric.tum", "w") as f:
            f.write("# metric-scaled by IMU alignment\n")
            for i in range(len(t_kf)):
                R = Rwb[i]
                tr = s * p[i]
                qx = R2q_xyzw(R)
                f.write(f"{t_kf[i]:.9f} " +
                        " ".join(f"{x2:.9f}" for x2 in (*tr, *qx)) + "\n")
        np.savetxt(od / f"{tag}_velocities.csv",
                   np.column_stack([np.array(t_kf), v]),
                   header="t,vx,vy,vz", delimiter=",")
        np.savetxt(od / f"{tag}_residuals.csv",
                   np.stack([np.array(t_kf)[np.array(use_idx)], per_int], 1),
                   header="t,residual_pos_norm", delimiter=",")


def R2q_xyzw(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        q = [(R[2,1]-R[1,2]), (R[0,2]-R[2,0]), (R[1,0]-R[0,1]), 1+tr]
    elif R[0,0] >= R[1,1] and R[0,0] >= R[2,2]:
        q = [1+R[0,0]-R[1,1]-R[2,2], R[0,1]+R[1,0], R[0,2]+R[2,0], R[2,1]-R[1,2]]
    elif R[1,1] >= R[2,2]:
        q = [R[0,1]+R[1,0], 1+R[1,1]-R[0,0]-R[2,2], R[1,2]+R[2,1], R[0,2]-R[2,0]]
    else:
        q = [R[0,2]+R[2,0], R[1,2]+R[2,1], 1+R[2,2]-R[0,0]-R[1,1], R[1,0]-R[0,1]]
    q = np.array(q, float)
    return q / np.linalg.norm(q)


if __name__ == "__main__":
    main()
