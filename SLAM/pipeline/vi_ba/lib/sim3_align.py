#!/usr/bin/env python3
"""Sim3-align one trajectory onto a metric reference (Umeyama).

This is how Project1 gave DROID its scale: rather than solving scale from the
IMU, stretch/rotate/translate the up-to-scale trajectory onto a trajectory that
is already metric (their VINS-Fusion or OpenVINS run). It makes the answer
depend on a second estimator being right first, which is why the default path
uses vi_align instead -- but it is the method that produced the delivered
DROID results, so it has to be reproducible for comparison.

Umeyama (1991) closed form: the similarity transform minimising squared error
between two point sets. evo does this too; implemented directly here so the
pipeline gains no dependency for one 20-line solve.

Poses are associated by nearest timestamp within --max-dt.

Usage: sim3_align.py <src.tum> <ref_metric.tum> <out.tum> [--max-dt 0.05]
                     [--lever-arm]
"""
import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation

R_BC = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
T_BC = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])


def umeyama(src, dst):
    """similarity transform mapping src -> dst; returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1                      # keep it a rotation, not a reflection
    R = U @ S @ Vt
    var = (s_c ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var)
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--max-dt", type=float, default=0.05)
    ap.add_argument("--lever-arm", action="store_true",
                    help="subtract the camera->IMU lever arm AFTER scaling "
                         "(only if src is camera-frame and already rotated)")
    a = ap.parse_args()

    S = np.loadtxt(a.src, comments="#")
    Rf = np.loadtxt(a.ref, comments="#")
    if S.ndim < 2 or Rf.ndim < 2:
        raise SystemExit("need >=2 poses in both files")

    # associate by nearest timestamp
    idx = np.abs(Rf[None, :, 0] - S[:, 0, None]).argmin(axis=1)
    dt = np.abs(Rf[idx, 0] - S[:, 0])
    ok = dt <= a.max_dt
    if ok.sum() < 20:
        raise SystemExit(
            f"only {ok.sum()} poses associate within {a.max_dt*1e3:.0f} ms "
            f"-- trajectories do not overlap")

    s, R, t = umeyama(S[ok, 1:4], Rf[idx[ok], 1:4])

    p = (s * (R @ S[:, 1:4].T)).T + t
    q = Rotation.from_matrix(
        R @ Rotation.from_quat(S[:, 4:8]).as_matrix()).as_quat()

    if a.lever_arm:
        # metric now, so the arm is meaningful -- same ordering rule as
        # apply_scale.py
        p = p - np.einsum("nij,j->ni", Rotation.from_quat(q).as_matrix(), T_BC)

    np.savetxt(a.out, np.column_stack([S[:, 0], p, q]), fmt="%.9f",
               header=f"Sim3-aligned to {a.ref}, scale {s:.6g}")
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    ref_seg = np.linalg.norm(np.diff(Rf[:, 1:4], axis=0), axis=1).sum()
    print(f"{len(p)} poses, {ok.sum()} associated, scale {s:.6g}, "
          f"path {seg:.2f} m (reference {ref_seg:.2f} m) -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
