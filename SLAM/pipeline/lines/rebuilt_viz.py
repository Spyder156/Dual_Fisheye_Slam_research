#!/usr/bin/env python
"""Draw the rebuilt line map the way it can actually be judged:
  - rebuilt_map.rrd : ONLY the lines (no trajectory, no points), line width in
    UI POINTS (readable at any zoom -- rerun: negative radius = ui points)
  - overlay PNG     : rebuilt segments projected onto a chosen frame, thick,
    to compare against the actual observed edges

usage: rebuilt_viz.py <rebuilt.csv> <rundir> --dataset D --config C --out DIR
                      [--t <timestamp s>]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).parent))
from debug_line_projection import load_cam  # noqa: E402
from doorway_overlay import project_kb4, load_mat, quat_R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rebuilt", type=Path)
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=float, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.atleast_1d(np.genfromtxt(args.rebuilt, delimiter=",", names=True))
    e1 = np.stack([d["x1"], d["y1"], d["z1"]], 1)
    e2 = np.stack([d["x2"], d["y2"], d["z2"]], 1)
    lid = d["lineid"].astype(int)

    # ---------------- 3D: lines only, readable width ----------------
    rr.init("rebuilt_lines", spawn=False)
    rr.save(str(args.out / "rebuilt_map.rrd"))
    rng = np.random.default_rng(5)
    cols = {t: rng.integers(70, 255, 3).tolist() for t in np.unique(lid)}
    rr.log("lines", rr.LineStrips3D(
        [np.stack([a, b]) for a, b in zip(e1, e2)],
        colors=[cols[t] for t in lid],
        radii=rr.Radius.ui_points(2.0)))
    print(f"{args.out}/rebuilt_map.rrd  ({len(e1)} segments, lines only)")

    # ---------------- 2D: overlay on a frame ----------------
    cfg = args.config.read_text()
    cam = load_cam(args.config, 0)
    T_b_c0 = load_mat(cfg, "IMU.T_b_c1")
    tr = np.loadtxt(args.rundir / "f_orb.txt")
    tf = tr[:, 0] / 1e9
    fr = np.genfromtxt(args.dataset / "frames.csv", delimiter=",", names=True)
    row = int(np.argmin(np.abs(tf - args.t))) if args.t else len(tf) // 3
    R_wb, p_wb = quat_R(tr[row, 4:8]), tr[row, 1:4]
    T_w_b = np.eye(4); T_w_b[:3, :3] = R_wb; T_w_b[:3, 3] = p_wb
    T = np.linalg.inv(T_b_c0) @ np.linalg.inv(T_w_b)

    # only lines actually OBSERVED near this time -- projecting the whole map
    # into one frame without occlusion is not a valid comparison
    ko = np.genfromtxt(args.rundir / "ml_orb_kfobs.csv", delimiter=",", names=True)
    near = set(ko["lineid"][np.abs(ko["t"] - tf[row]) < 2.0].astype(int))

    fi = int(np.argmin(np.abs(fr["t"] - tf[row])))
    img = cv2.imread(str(args.dataset / "cam0" / f"{int(fr['frame'][fi]):06d}.jpg"))
    img = cv2.rotate(img, cv2.ROTATE_180)
    h, w = img.shape[:2]
    ndrawn = 0
    for a, b, t in zip(e1, e2, lid):
        if t not in near:
            continue
        S = np.linspace(0, 1, 32)[:, None]
        Pw = (1 - S) * a + S * b
        Pc = (T[:3, :3] @ Pw.T).T + T[:3, 3]
        ang = np.arctan2(np.hypot(Pc[:, 0], Pc[:, 1]), Pc[:, 2])
        ok = ang < np.radians(92)
        if ok.sum() < 6:
            continue
        uv = project_kb4(Pc[ok], cam)
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if ins.sum() < 6:
            continue
        pts = uv[ins]
        pts = np.stack([w - 1 - pts[:, 0], h - 1 - pts[:, 1]], -1).astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, cols[t], 3, cv2.LINE_AA)
        ndrawn += 1
    p = args.out / f"rebuilt_overlay_t{tf[row]:.2f}.png"
    cv2.imwrite(str(p), img)
    print(f"{p}  ({ndrawn} rebuilt segments projected)")


if __name__ == "__main__":
    main()
