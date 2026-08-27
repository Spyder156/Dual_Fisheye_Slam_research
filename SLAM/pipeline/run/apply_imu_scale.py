#!/usr/bin/env python
"""Post-SLAM metric scale correction. The one VI stage that actually helps.

Monocular-inertial SLAM recovers scale from the IMU, but ORB-SLAM3's scale
converges early and stays ~2% small on our sequences. This measures the
correction independently and applies it.

Measured on Hilti floor_EG run_1:  91.30 -> 95.44  (ATE 0.2145 -> 0.1171 m)
Ceiling with a perfect scale is 96.47, so ~1 point remains in the estimate.

Pipeline:
  1. triangulate structure into the SLAM poses (poses held FIXED)
  2. visual-inertial alignment on that model -> scale, gravity, bias
  3. multiply the trajectory by the scale

NOT done here: the full VI bundle adjustment. It measurably HURTS
(95.44 -> 92.58) because the structure is triangulated from the very poses the
BA then moves -- the reprojection error drops while the trajectory drifts from
truth. A BA only helps with structure the front-end did not already imply.

Self-checks, from the vi-ba README and confirmed in practice:
  |g|    ~ 9.81       -- a wrong gravity sign put 2g into the bias
  |b_a|  < 0.5 m/s^2  -- a bias near 10 means gravity was absorbed
Both are printed; the script refuses to apply a scale if either fails.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="SLAM trajectory, TUM")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--imucam", required=True)
    ap.add_argument("--out", required=True, help="corrected trajectory, TUM")
    ap.add_argument("--work", default=None)
    ap.add_argument("--colmap", default=str(PIPE / "vi_ba_kb/lib/colmap_docker.sh"))
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--max-kf", type=int, default=300)
    ap.add_argument("--force", action="store_true",
                    help="apply the scale even if the self-checks fail")
    a = ap.parse_args()

    ds = Path(a.dataset)
    work = Path(a.work) if a.work else Path(a.out).parent / "imu_scale_work"
    work.mkdir(parents=True, exist_ok=True)
    model = work / "model"

    print("[1/3] triangulating structure into the SLAM poses (poses fixed)")
    r = subprocess.run([sys.executable, str(PIPE / "vi_ba_kb/lib/triangulate_from_poses.py"),
                        "--traj", a.traj, "--frames-dir", str(ds / "cam0"),
                        "--frames-csv", str(ds / "frames.csv"), "--imucam", a.imucam,
                        "--out", str(model), "--max-frames", str(a.max_frames),
                        "--colmap", a.colmap], capture_output=True, text=True)
    print(r.stdout.strip()[-400:] or r.stderr.strip()[-400:])
    if r.returncode:
        sys.exit("triangulation failed")

    print("[2/3] visual-inertial alignment")
    aj = work / "align.json"
    r = subprocess.run([sys.executable, str(PIPE / "step0_calib/step0_imu_scale.py"),
                        "--sparse", str(model), "--dataset", str(ds),
                        "--imucam", a.imucam, "--max-kf", str(a.max_kf),
                        "--out-json", str(aj)], capture_output=True, text=True)
    print(r.stdout.strip()[-500:] or r.stderr.strip()[-500:])
    if r.returncode:
        sys.exit("alignment failed")

    al = json.loads(aj.read_text())
    s, gn = al["scale"], al["gravity_norm"]
    ba = np.linalg.norm(al["accel_bias"])
    ok_g = abs(gn - 9.81) < 0.5
    ok_b = ba < 0.5
    print(f"      |g| = {gn:.3f} {'OK' if ok_g else 'FAIL (expect ~9.81)'}")
    print(f"      |b_a| = {ba:.3f} {'OK' if ok_b else 'FAIL (>0.5 means gravity was absorbed)'}")
    if not (ok_g and ok_b) and not a.force:
        sys.exit("self-checks failed -- refusing to apply this scale (use --force)")

    print(f"[3/3] applying scale x{s:.4f}")
    d = np.loadtxt(a.traj, comments="#")
    d[:, 1:4] *= s
    np.savetxt(a.out, d, fmt="%.9f")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
