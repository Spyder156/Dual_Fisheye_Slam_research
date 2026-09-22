#!/usr/bin/env python
"""Score a trajectory against Hilti-Trimble ground truth.

Protocol (what the challenge actually does, not what flatters us):
  - GT poses are cam0; an inertial estimate is BODY poses. --config supplies
    the settings yaml with IMU.T_b_c1 so the lever arm is applied per pose:
        p_w_c0 = p_w_b + R_wb @ t_b_c0        # t_b_c0: cam0 origin in body
  - A GT timestamp is COVERED only if it falls between two estimate samples
    whose spacing is <= --max-gap AND its nearest sample is <= --max-dt away.
    Interpolation never bridges a tracking hole: missing stays missing.
  - Alignment: SE(3) Umeyama on the covered pairs (challenge protocol).
  - SCORE(strict) = sum(100 * exp(-C * e_i)) / N_gt  -- an uncovered GT pose
    contributes 0. The challenge additionally REJECTS a submission with
    coverage < 99%; that verdict is printed, the number is still reported.

Usage: hilti_score.py <traj> <groundtruth.txt> [--config settings.yaml]
  traj : traj.csv (t,px,py,pz,qx,qy,qz,qw header, t seconds)
         or f_orb.txt (TUM, space-separated, no header, t NANOSECONDS)
  gt   : TUM  # timestamp tx ty tz qx qy qz qw   (t seconds, cam0 frame)
"""
import argparse
import re
from pathlib import Path

import numpy as np

C = 0.46051701859880917       # 100*exp(-C*1m) = 63.1, the challenge constant
COVERAGE_REQUIRED = 99.0      # challenge rejection rule [%]


def umeyama_se3(A, B):
    """Rigid transform aligning A onto B (no scale). Returns (R, t)."""
    mA, mB = A.mean(0), B.mean(0)
    H = (A - mA).T @ (B - mB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mB - R @ mA


def load_traj(path):
    """Estimate trajectory: (te [s], Pe Nx3 [m], Qe Nx4 xyzw or None).

    Sniffs the two formats we produce: comma = traj.csv with header (t already
    seconds); whitespace = f_orb.txt straight out of the estimator, t in int64
    NANOSECONDS on the dataset clock (same clock as GT)."""
    with open(path) as f:
        first = f.readline()
    if "," in first:
        d = np.genfromtxt(path, delimiter=",", names=True)
        te = np.atleast_1d(d["t"])
        Pe = np.stack([d["px"], d["py"], d["pz"]], 1)
        Qe = np.stack([d["qx"], d["qy"], d["qz"], d["qw"]], 1)
    else:
        a = np.loadtxt(path)
        te, Pe, Qe = a[:, 0], a[:, 1:4], a[:, 4:8]
    if te[0] > 1e12:  # nanoseconds -> seconds
        te = te / 1e9
    return te, Pe, Qe


def load_T_b_c1(yaml_path):
    """IMU.T_b_c1 from an ORB-SLAM3 settings yaml: 4x4, cam0 -> body
    (p_b = T_b_c1 @ p_c), row-major data list."""
    txt = Path(yaml_path).read_text()
    m = re.search(r"IMU\.T_b_c1:.*?data:\s*\[(.*?)\]", txt, re.S)
    if not m:
        return None
    v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v).reshape(4, 4)


def quat_to_R(q):
    """xyzw quaternions (N,4) -> rotation matrices (N,3,3). Active, R_wb."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)], -1),
        np.stack([2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
        np.stack([2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)], -1),
    ], 1)


def coverage_mask(te, tg, max_dt, max_gap):
    """Which GT timestamps have a legitimate estimate: bracketed by estimate
    samples closer than max_gap apart, with the nearest sample within max_dt.
    Everything else is a HOLE and must stay one."""
    idx = np.searchsorted(te, tg)                # te[idx-1] <= tg < te[idx]
    inside = (idx > 0) & (idx < len(te))
    ok = np.zeros(len(tg), bool)
    ii = idx[inside]
    gap = te[ii] - te[ii - 1]
    near = np.minimum(tg[inside] - te[ii - 1], te[ii] - tg[inside])
    ok[inside] = (gap <= max_gap) & (near <= max_dt)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--max-dt", type=float, default=0.02,
                    help="max distance to the nearest estimate sample [s]")
    ap.add_argument("--max-gap", type=float, default=0.25,
                    help="max estimate-sample spacing to interpolate across [s]")
    ap.add_argument("--config", type=Path, default=None,
                    help="ORB settings yaml with IMU.T_b_c1: scores cam0 "
                         "positions instead of body positions")
    args = ap.parse_args()

    te, Pe, Qe = load_traj(args.traj)
    order = np.argsort(te)
    te, Pe, Qe = te[order], Pe[order], Qe[order]

    if args.config is not None:
        T_b_c = load_T_b_c1(args.config)
        if T_b_c is None:
            raise SystemExit(f"no IMU.T_b_c1 in {args.config}")
        # p_w_c0 = p_w_b + R_wb @ t_b_c0
        Pe = Pe + np.einsum("nij,j->ni", quat_to_R(Qe), T_b_c[:3, 3])
        print(f"body -> cam0 lever arm applied: |t| = {np.linalg.norm(T_b_c[:3,3])*100:.1f} cm")
    else:
        print("WARNING: scoring raw (body) positions; GT is cam0 -- pass --config")

    g = np.loadtxt(args.gt)
    tg, Pg = g[:, 0], g[:, 1:4]

    ok = coverage_mask(te, tg, args.max_dt, args.max_gap)
    if ok.sum() < 10:
        raise SystemExit(f"only {ok.sum()} GT poses covered -- nothing to score")

    A = np.stack([np.interp(tg[ok], te, Pe[:, i]) for i in range(3)], 1)
    B = Pg[ok]
    R, t = umeyama_se3(A, B)
    err = np.linalg.norm((R @ A.T).T + t - B, axis=1)

    rmse = float(np.sqrt((err ** 2).mean()))
    cov = 100.0 * ok.sum() / len(tg)
    score_cov = float((100 * np.exp(-C * err)).mean())      # covered poses only
    score_strict = float((100 * np.exp(-C * err)).sum() / len(tg))  # holes = 0

    print(f"covered      : {ok.sum()}/{len(tg)} GT poses  (coverage {cov:.2f}%)")
    print(f"ATE RMSE     : {rmse:.4f} m   (covered poses)")
    print(f"  mean/med   : {err.mean():.4f} / {np.median(err):.4f} m")
    print(f"  max/min    : {err.max():.4f} / {err.min():.4f} m")
    print(f"score covered : {score_cov:.2f}   (uncovered ignored -- diagnostic only)")
    print(f"SCORE strict  : {score_strict:.2f} / 100   (uncovered = 0)")
    verdict = "ACCEPTED" if cov >= COVERAGE_REQUIRED else "REJECTED (<99% coverage)"
    print(f"protocol      : {verdict}")
    # machine-readable line for the N-run harness (nrun.sh)
    print(f"CSV {score_strict:.2f},{rmse:.4f},{cov:.2f}")


if __name__ == "__main__":
    main()
