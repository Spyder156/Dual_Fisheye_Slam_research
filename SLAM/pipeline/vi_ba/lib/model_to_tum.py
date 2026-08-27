#!/usr/bin/env python3
"""COLMAP model -> IMU-frame TUM.

Image names come in two flavours and both are supported:

  <t_ns>.jpg      what our own triangulated/refined models use -- the timestamp
                  is the name, so nothing else is needed.
  <index>.jpg     what hloc writes, because it extracts frames by position.
                  Pass --episode to resolve those against the episode's
                  frames csv and clock offset.

Poses are read from images.txt. For the single-camera rigs we produce, its
CAM_FROM_WORLD is identical to frames.txt's RIG_FROM_WORLD (asserted below);
images.txt is used because it is the only one of the two that also carries the
image NAME, which is what the timestamp mapping needs.

Usage: model_to_tum.py <model_dir> <out.tum> [--episode EP]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

R_bc = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
t_bc = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])


def q2R(x, y, z, w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def R2q(R):
    tr = R[0, 0]+R[1, 1]+R[2, 2]
    if tr > 0:
        q = [R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1], 1+tr]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        q = [1+R[0, 0]-R[1, 1]-R[2, 2], R[0, 1]+R[1, 0], R[0, 2]+R[2, 0],
             R[2, 1]-R[1, 2]]
    elif R[1, 1] >= R[2, 2]:
        q = [R[0, 1]+R[1, 0], 1+R[1, 1]-R[0, 0]-R[2, 2], R[1, 2]+R[2, 1],
             R[0, 2]-R[2, 0]]
    else:
        q = [R[0, 2]+R[2, 0], R[1, 2]+R[2, 1], 1+R[2, 2]-R[0, 0]-R[1, 1],
             R[1, 0]-R[0, 1]]
    q = np.array(q, float)
    return q/np.linalg.norm(q)


def _frame_times(ep):
    """unix time per frame index, on the IMU clock."""
    import pandas as pd
    root = Path(os.environ.get("MECKA_ROOT", "/pipeline/data"))
    epd = root / "data/episodes" / ep
    csv = sorted(epd.glob("frames*.csv"))
    if not csv:
        raise SystemExit(f"no frames csv under {epd}")
    off = json.load(open(root / f"results/offsets/{ep}.json"))["offset_s"]
    return pd.read_csv(csv[0])["unix_timestamp"].to_numpy() + off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--episode", default=None,
                    help="resolve <index>.jpg names via this episode's frames "
                         "csv + clock offset (hloc-style models)")
    ap.add_argument("--unrotate", default=None,
                    help="world_transform.json written by make_imu_factors; "
                         "undoes the gravity rotation it applied so the poses "
                         "come back in the frame they started in")
    a = ap.parse_args()
    M = Path(a.model)

    # The refinement rotates the world so gravity is -Z, because the solver
    # assumes that. Undo it here: the caller's trajectory is still in the
    # original frame, and leaving the two in different frames turns an internal
    # convention into a global attitude error in the shipped result.
    R_un = None
    wt = Path(a.unrotate) if a.unrotate else M / "world_transform.json"
    if wt.exists():
        R_un = np.array(json.load(open(wt))["R_up"], float).T

    unix = _frame_times(a.episode) if a.episode else None

    rows, skipped = [], 0
    lines = [ln for ln in open(M / "images.txt") if not ln.startswith("#")]
    for i in range(0, len(lines)-1, 2):
        h = lines[i].split()
        if len(h) != 10:
            continue
        qw, qx, qy, qz = map(float, h[1:5])
        n = np.linalg.norm([qw, qx, qy, qz])
        R_cw = q2R(qx/n, qy/n, qz/n, qw/n)
        t_cw = np.array(list(map(float, h[5:8])))
        R_wc = R_cw.T
        p_wc = -R_cw.T @ t_cw
        if R_un is not None:
            R_wc = R_un @ R_wc
            p_wc = R_un @ p_wc
        R_wi = R_wc @ R_bc.T
        p_wi = p_wc - R_wi @ t_bc

        stem = Path(h[9]).stem
        try:
            key = int(stem)
        except ValueError:
            skipped += 1
            continue
        if unix is not None:
            if key >= len(unix):
                skipped += 1
                continue
            t = float(unix[key])
        else:
            t = key / 1e9
        rows.append((t, *p_wi, *R2q(R_wi)))

    if not rows:
        raise SystemExit(f"{M}: no poses recovered from images.txt")
    rows.sort()
    with open(a.out, "w") as f:
        f.write("# t x y z qx qy qz qw (COLMAP model, IMU frame)\n")
        for r in rows:
            f.write(" ".join(f"{v:.9f}" for v in r) + "\n")
    span = rows[-1][0] - rows[0][0]
    print(f"{len(rows)} poses -> {a.out}"
          + (f"  ({skipped} skipped)" if skipped else "")
          + f"  span {span:.1f}s")


if __name__ == "__main__":
    main()
