#!/usr/bin/env python
"""Score a trajectory against Hilti-Trimble ground truth.

Metrics: ATE RMSE after SE(3) Umeyama alignment (challenge protocol), coverage,
and the official challenge score  S = mean(100 * exp(-0.46051701859880917 * e_i)).

Usage: hilti_score.py <traj> <groundtruth.txt>
  traj : either traj.csv (t,px,py,pz,qx,qy,qz,qw header, t in seconds)
         or f_orb.txt    (TUM, space-separated, no header, t in NANOSECONDS)
  gt.txt : TUM  # timestamp tx ty tz qx qy qz qw   (t in seconds)
"""
import argparse
from pathlib import Path

import numpy as np

C = 0.46051701859880917


def umeyama_se3(A, B):
    """Rigid transform aligning A onto B (no scale). Returns (R, t)."""
    mA, mB = A.mean(0), B.mean(0)
    H = (A - mA).T @ (B - mB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mB - R @ mA


def load_traj(path):
    """Estimate positions (te [s], Pe Nx3 [m], world frame of the estimator).

    Sniffs the two formats we produce: comma = traj.csv with header (t already
    seconds); whitespace = f_orb.txt straight out of the estimator, t in int64
    NANOSECONDS on the dataset clock (same clock as GT)."""
    with open(path) as f:
        first = f.readline()
    if "," in first:
        d = np.genfromtxt(path, delimiter=",", names=True)
        te = np.atleast_1d(d["t"])
        Pe = np.stack([d["px"], d["py"], d["pz"]], 1)
    else:
        a = np.loadtxt(path)
        te, Pe = a[:, 0], a[:, 1:4]
    if te[0] > 1e12:  # nanoseconds -> seconds
        te = te / 1e9
    return te, Pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--max-dt", type=float, default=0.02,
                    help="max timestamp difference to accept a match [s]")
    args = ap.parse_args()

    te, Pe = load_traj(args.traj)

    g = np.loadtxt(args.gt)
    tg, Pg = g[:, 0], g[:, 1:4]

    # Interpolate the estimate to the GT timestamps. GT sits on a different phase
    # grid than the camera frames, so nearest-neighbour matching throws away most
    # of the trajectory. Interpolation is the standard evaluation and is what the
    # challenge evaluator effectively does on a submitted trajectory.
    inside = (tg >= te[0]) & (tg <= te[-1])
    if inside.sum() < 10:
        raise SystemExit(f"est span [{te[0]:.2f},{te[-1]:.2f}] barely overlaps gt "
                         f"[{tg[0]:.2f},{tg[-1]:.2f}]")
    A = np.stack([np.interp(tg[inside], te, Pe[:, i]) for i in range(3)], 1)
    B = Pg[inside]
    ok = inside
    R, t = umeyama_se3(A, B)
    err = np.linalg.norm((R @ A.T).T + t - B, axis=1)

    rmse = float(np.sqrt((err ** 2).mean()))
    score = float((100 * np.exp(-C * err)).mean())
    cov = 100.0 * ok.sum() / len(tg)

    print(f"matched      : {ok.sum()}/{len(tg)} GT poses  (coverage {cov:.2f}%)")
    print(f"ATE RMSE     : {rmse:.4f} m")
    print(f"  mean/med   : {err.mean():.4f} / {np.median(err):.4f} m")
    print(f"  max/min    : {err.max():.4f} / {err.min():.4f} m")
    print(f"CHALLENGE SCORE : {score:.2f} / 100")
    # machine-readable line for the N-run harness (nrun.sh)
    print(f"CSV {score:.2f},{rmse:.4f},{cov:.2f}")


if __name__ == "__main__":
    main()
