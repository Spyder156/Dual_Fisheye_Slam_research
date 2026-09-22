#!/usr/bin/env python
"""Targeted tests for hilti_score.py -- each one has a known answer.

1. LEVER ARM. A body spinning in place with cam0 on a 10 cm arm: GT (cam0) is
   a circle, the body is a point. Scored raw, the RMSE must be ~the arm radius
   (rotation cannot be absorbed by a rigid alignment of a point onto a circle);
   with the lever arm applied it must be ~0.
2. HOLES STAY HOLES. An estimate with a 10 s gap must not cover GT inside the
   gap: coverage < 99% -> protocol REJECTED, and the strict score is scaled
   down by the missing fraction while the covered score is not.
"""
import subprocess
import sys
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SCORE = HERE / "hilti_score.py"
TMP = Path("/tmp/claude-1000/-home-raghav-workspace-INSV-STITCHING/3d7e92da-d72b-4eb8-b5c2-7050abc3ca19/scratchpad/score_test")
TMP.mkdir(parents=True, exist_ok=True)


def run(traj, gt, *extra):
    out = subprocess.run([sys.executable, str(SCORE), str(traj), str(gt), *extra],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout, out.stderr)
        raise SystemExit("scorer crashed")
    txt = out.stdout

    def g(pat):
        return float(re.search(pat, txt).group(1))
    return dict(cov=g(r"coverage ([\d.]+)%"), rmse=g(r"ATE RMSE\s+: ([\d.]+)"),
                strict=g(r"SCORE strict\s+: ([\d.]+)"),
                covered=g(r"score covered : ([\d.]+)"),
                rejected="REJECTED" in txt)


def write_tum(path, t, P, Q):
    np.savetxt(path, np.column_stack([t, P, Q]), fmt="%.9f")


# ---------------------------------------------------------------- lever arm --
# body: fixed at origin, spinning about z at 1 rad/s. cam0 at +10 cm body x.
t = np.arange(0, 30, 1 / 30)
half = np.sin(t / 2)
Q = np.stack([np.zeros_like(t), np.zeros_like(t), half, np.cos(t / 2)], 1)  # xyzw about z
P_body = np.zeros((len(t), 3))
r = 0.10
P_cam = np.stack([r * np.cos(t), r * np.sin(t), np.zeros_like(t)], 1)  # R_wb@[r,0,0]

est = TMP / "est_body.txt"; write_tum(est, t, P_body, Q)
gt = TMP / "gt_cam0.txt";   write_tum(gt, t, P_cam, Q)
cfg = TMP / "cfg.yaml"
cfg.write_text("IMU.T_b_c1: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n"
               "   data: [1,0,0,0.10, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n")

raw = run(est, gt)
fix = run(est, gt, "--config", cfg)
# a point aligned onto a circle of radius r: best rigid fit leaves rms ~= r
assert abs(raw["rmse"] - r) < 0.02, f"raw body scoring should miss by ~{r}m, got {raw['rmse']}"
assert fix["rmse"] < 1e-6, f"lever-arm-corrected rmse should be ~0, got {fix['rmse']}"
print(f"PASS lever arm: raw rmse {raw['rmse']:.3f} ~= {r}; corrected {fix['rmse']:.2e}")

# ------------------------------------------------------------- holes stay ----
# estimate = GT with a 10 s hole in the middle (frames 300..600 of 900 dropped)
keep = np.ones(len(t), bool); keep[300:600] = False
est2 = TMP / "est_hole.txt"; write_tum(est2, t[keep], P_body[keep], Q[keep])
hole = run(est2, gt, "--config", cfg)
expected_cov = 100.0 * keep.sum() / len(t)
assert abs(hole["cov"] - expected_cov) < 1.0, f"coverage {hole['cov']} != {expected_cov:.1f}"
assert hole["rejected"], "sub-99% coverage must be REJECTED"
assert hole["rmse"] < 1e-6, "covered part is exact"
assert abs(hole["strict"] - hole["covered"] * keep.sum() / len(t)) < 0.5, \
    "strict score must scale with the missing fraction"
print(f"PASS holes: coverage {hole['cov']:.1f}% (exp {expected_cov:.1f}), REJECTED, "
      f"strict {hole['strict']:.1f} vs covered {hole['covered']:.1f}")

print("ALL PASS")
