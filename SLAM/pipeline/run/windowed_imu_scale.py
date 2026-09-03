#!/usr/bin/env python
"""Time-varying metric scale correction.

WHY: a single global scale corrects the MEAN of a drifting error. Measured
scale (estimate/GT path length) over our three sequences:

    sequence    first 25%   last 25%
    run_1         0.985       0.980
    run_2         0.992       1.683     <-- one scalar cannot fix this
    10-16         1.018       1.002

The global correction gave +4.12 on run_1 (error nearly constant) and ~0 on the
other two. So scale must be estimated as a FUNCTION OF TIME.

HOW:
  1. Global visual-inertial solve for gravity g and accel bias ba. These are
     genuinely constant, so they get all the data.
  2. Per-window solve for scale ONLY, with g and ba held fixed. Unknowns per
     window are s plus the window's velocities.
  3. Re-integrate the trajectory scaling each DISPLACEMENT by its local scale:
         P'(t_k) = P'(t_0) + sum_i s(t_i) * (P(t_i) - P(t_{i-1}))
     Multiplying absolute positions by a varying scale would warp the shape
     about an arbitrary origin; scaling increments corrects drift while leaving
     local geometry intact.

Self-checks (same as the global stage): |g| ~ 9.81, |b_a| < 0.5.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "step0_calib"))
from step0_imu_scale import (G, preintegrate, q2R, read_images_bin)  # noqa: E402


def load_poses(sparse, dataset, imucam_R, imucam_t, max_kf):
    imgs = read_images_bin(Path(sparse) / "images.bin")
    fr = np.genfromtxt(Path(dataset) / "frames.csv", delimiter=",", names=True)
    tmap = dict(zip(np.atleast_1d(fr["frame"]).astype(int), np.atleast_1d(fr["t"])))
    rows = []
    for n, (q, t) in imgs.items():
        if "/" in n and not n.startswith("cam0/"):
            continue
        k = int(n.split("/")[-1].split(".")[0])
        if k not in tmap:
            continue
        R = q2R(q)
        Rcw, C = R.T, -R.T @ t
        R_WI = Rcw @ imucam_R.T
        rows.append((tmap[k], C - R_WI @ imucam_t, R_WI))
    rows.sort(key=lambda r: r[0])
    rows = rows[:max_kf]
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def build_rows(T, P, Rw, ti, acc, gyr, i0, i1):
    """Preintegrated interval rows for keyframes [i0, i1)."""
    out = []
    for i in range(i0, i1 - 1):
        pi = preintegrate(ti, acc, gyr, T[i], T[i + 1])
        if pi is None:
            continue
        dP, dV, Jp, Jv, dt = pi
        if dt <= 0:
            continue
        out.append((i, dP, dV, Jp, Jv, dt))
    return out


def solve_window(rows, T, P, Rw, g, ba, i0, i1):
    """Scale only, gravity and bias fixed. Unknowns: s, then v per keyframe."""
    n = i1 - i0
    if len(rows) < 3:
        return None
    nx = 1 + 3 * n
    A, b = [], []
    for (i, dP, dV, Jp, Jv, dt) in rows:
        k = i - i0
        Ri = Rw[i]
        r = np.zeros((3, nx))
        r[:, 0:1] = (P[i + 1] - P[i]).reshape(3, 1)
        r[:, 1 + 3*k: 4 + 3*k] = -np.eye(3) * dt
        A.append(r); b.append(Ri @ (dP - Jp @ ba) + 0.5 * g * dt * dt)
        r = np.zeros((3, nx))
        r[:, 1 + 3*k: 4 + 3*k] = -np.eye(3)
        if 4 + 3*k + 3 <= nx:
            r[:, 4 + 3*k: 7 + 3*k] = np.eye(3)
        A.append(r); b.append(Ri @ (dV - Jv @ ba) + g * dt)
    A = np.vstack(A); b = np.concatenate(b)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--sparse", required=True, help="triangulated model dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--imucam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align-json", default=None, help="global align.json (g, ba)")
    ap.add_argument("--window", type=int, default=40, help="keyframes per window")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-kf", type=int, default=400)
    ap.add_argument("--smooth", type=int, default=3, help="median filter width")
    ap.add_argument("--clamp", type=float, default=0.35,
                    help="reject window scales further than this from the median")
    a = ap.parse_args()

    import re
    blk = re.split(r"\ncam\d+:", Path(a.imucam).read_text())[1]
    Tm = np.array([float(x) for x in re.findall(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", blk.split("cam_overlaps")[0])]).reshape(4, 4)
    R_CtoI, p_CinI = Tm[:3, :3], Tm[:3, 3]

    T, P, Rw = load_poses(a.sparse, a.dataset, R_CtoI, p_CinI, a.max_kf)
    print(f"{len(T)} keyframes, {T[-1]-T[0]:.1f} s")

    d = np.genfromtxt(Path(a.dataset) / "imu.csv", delimiter=",", names=True)
    ti = d["t"]
    gyr = np.stack([d["gx"], d["gy"], d["gz"]], 1)
    acc = np.stack([d["ax"], d["ay"], d["az"]], 1)

    al = json.loads(Path(a.align_json).read_text())
    # align.json stores gravity_dir pointing DOWN (negated at source); the solve
    # here uses the same convention as step0_imu_scale's internal g.
    g = -np.array(al["gravity_dir"], float) * al["gravity_norm"]
    ba = np.array(al["accel_bias"], float)
    print(f"global: |g| {np.linalg.norm(g):.3f}  |b_a| {np.linalg.norm(ba):.3f}")
    if abs(np.linalg.norm(g) - 9.81) > 0.5 or np.linalg.norm(ba) > 0.5:
        sys.exit("global self-checks failed -- refusing to build a windowed scale")

    allrows = build_rows(T, P, Rw, ti, acc, gyr, 0, len(T))
    by_i = {r[0]: r for r in allrows}

    centres, scales = [], []
    for i0 in range(0, len(T) - a.window, a.stride):
        i1 = i0 + a.window
        rows = [by_i[i] for i in range(i0, i1 - 1) if i in by_i]
        s = solve_window(rows, T, P, Rw, g, ba, i0, i1)
        if s is None or not np.isfinite(s) or s <= 0:
            continue
        centres.append(0.5 * (T[i0] + T[i1 - 1])); scales.append(s)
    centres, scales = np.array(centres), np.array(scales)
    if len(scales) < 3:
        sys.exit("too few valid windows")

    med = np.median(scales)
    keep = np.abs(scales - med) <= a.clamp * med
    print(f"{len(scales)} windows, {int((~keep).sum())} rejected as outliers "
          f"(> {a.clamp*100:.0f}% from median {med:.4f})")
    centres, scales = centres[keep], scales[keep]

    if a.smooth > 1 and len(scales) >= a.smooth:
        sm = np.copy(scales)
        h = a.smooth // 2
        for i in range(len(scales)):
            sm[i] = np.median(scales[max(0, i-h): i+h+1])
        scales = sm
    print(f"scale over time: min {scales.min():.4f}  max {scales.max():.4f}  "
          f"median {np.median(scales):.4f}  drift {scales[-1]-scales[0]:+.4f}")

    # ---- re-integrate, scaling each DISPLACEMENT by its local scale ---------
    traj = np.loadtxt(a.traj, comments="#")
    tt = traj[:, 0] / (1e9 if traj[0, 0] > 1e12 else 1.0)
    s_of_t = np.interp(tt, centres, scales, left=scales[0], right=scales[-1])
    Pin = traj[:, 1:4]
    d_ = np.diff(Pin, axis=0)
    Pout = np.zeros_like(Pin)
    Pout[0] = Pin[0] * np.median(scales)
    Pout[1:] = Pout[0] + np.cumsum(d_ * s_of_t[1:, None], axis=0)
    traj[:, 1:4] = Pout
    np.savetxt(a.out, traj, fmt="%.9f")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
