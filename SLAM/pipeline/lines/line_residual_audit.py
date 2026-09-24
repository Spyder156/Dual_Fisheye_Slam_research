#!/usr/bin/env python
"""Line-vs-observation residual audit (the "line 3667" metric, whole map).

For every map line: worst angular distance [deg] of any observed endpoint
bearing from the line's predicted great circle, across ALL its keyframe
observations. A healthy landmark explains what created it (<~0.5 deg); a
corrupted one (bad merge normal, stale geometry) misses by degrees.

Inputs are the estimator's OWN dumps (run dir): ml_orb.csv (world endpoints),
ml_orb_kfobs.csv (pixel endpoints per KF per cam), ml_orb_kfpose_b0.csv
(T_c_b0: world->camera per KF per lens). Pixels are lifted with the verified
spherical KB inverse (debug_line_projection.unproject_kb4), NOT the estimator's
z=1 inverse, so this audit is independent of defect #1.

Usage: line_residual_audit.py <rundir> --config <settings.yaml> [--min-obs 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from debug_line_projection import load_cam, unproject_kb4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--min-obs", type=int, default=2,
                    help="audit lines with at least this many dumped obs")
    args = ap.parse_args()

    cams = [load_cam(args.config, 0), load_cam(args.config, 1)]

    ml = np.genfromtxt(args.rundir / "ml_orb.csv", delimiter=",", names=True)
    obs = np.genfromtxt(args.rundir / "ml_orb_kfobs.csv", delimiter=",", names=True)
    pos = np.genfromtxt(args.rundir / "ml_orb_kfpose_b0.csv", delimiter=",", names=True)

    # T_c_b0 per (kfid, cam): world(b0) -> camera, row-major R then t
    P = {}
    for r in pos:
        R = np.array([[r["r00"], r["r01"], r["r02"]],
                      [r["r10"], r["r11"], r["r12"]],
                      [r["r20"], r["r21"], r["r22"]]])
        P[(int(r["kfid"]), int(r["cam"]))] = (R, np.array([r["tx"], r["ty"], r["tz"]]))

    # line geometry: d_w unit, m_w = e1 x d_w  (moment about world origin)
    G = {}
    for r in np.atleast_1d(ml):
        e1 = np.array([r["x1"], r["y1"], r["z1"]])
        e2 = np.array([r["x2"], r["y2"], r["z2"]])
        d = e2 - e1
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d /= n
        G[int(r["id"])] = (d, np.cross(e1, d))

    worst = {}   # lineid -> max residual [rad]
    nobs = {}
    for r in np.atleast_1d(obs):
        lid = int(r["lineid"])
        key = (int(r["kfid"]), int(r["cam"]))
        if lid not in G or key not in P:
            continue
        (d_w, m_w), (R, t) = G[lid], P[key]
        d_c = R @ d_w
        m_c = R @ m_w + np.cross(t, d_c)
        nm = np.linalg.norm(m_c)
        if nm < 1e-9:
            continue
        n_c = m_c / nm
        b = unproject_kb4(np.array([r["x1"], r["x2"]]),
                          np.array([r["y1"], r["y2"]]), cams[int(r["cam"])])
        res = float(np.max(np.arcsin(np.minimum(1.0, np.abs(b @ n_c)))))
        worst[lid] = max(worst.get(lid, 0.0), res)
        nobs[lid] = nobs.get(lid, 0) + 1

    v = np.degrees([w for l, w in worst.items() if nobs[l] >= args.min_obs])
    v = np.sort(np.asarray(v))
    if not len(v):
        raise SystemExit("no lines with enough observations")
    print(f"lines audited : {len(v)} (>= {args.min_obs} obs)")
    print(f"worst-obs residual [deg]: median {np.median(v):.3f}  "
          f"p90 {np.percentile(v, 90):.3f}  max {v[-1]:.2f}")
    for thr in (0.5, 1.0, 2.0, 5.0):
        print(f"  > {thr:3.1f} deg : {(v > thr).mean() * 100:5.1f} %")


if __name__ == "__main__":
    main()
