#!/usr/bin/env python
"""Keypoint + match visualization between fisheye frame pairs.

For each requested frame index: detect corners in frame i (inside the valid
circle), KLT-track them into frame i+gap, and render the pair side by side with
match lines (green = tracked, red dot = lost). Run with the `vision` env python.

Usage: viz_matches.py <dataset_dir> <out_dir> [--frames 200 900 1600 2200] [--gap 1]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path


def match_pair(dsdir: Path, i: int, gap: int, cam: str):
    a = cv2.imread(str(dsdir / cam / f"{i:06d}.jpg"), 0)
    b = cv2.imread(str(dsdir / cam / f"{i+gap:06d}.jpg"), 0)
    if a is None or b is None:
        return None
    a = cv2.resize(a, None, fx=0.5, fy=0.5)
    b = cv2.resize(b, None, fx=0.5, fy=0.5)
    h, w = a.shape
    mask = np.zeros_like(a)
    cv2.circle(mask, (w // 2, h // 2), int(0.46 * w), 255, -1)
    pts = cv2.goodFeaturesToTrack(a, 400, 0.01, 15, mask=mask)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=5)
    st = st.ravel().astype(bool)
    va = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    vb = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    canvas = np.concatenate([va, vb], axis=1)
    for p0, p1i, ok in zip(pts.reshape(-1, 2), p1.reshape(-1, 2), st):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1i[0]) + w, int(p1i[1])
        if ok:
            cv2.circle(canvas, (x0, y0), 3, (80, 220, 80), -1)
            cv2.circle(canvas, (x1, y1), 3, (80, 220, 80), -1)
            cv2.line(canvas, (x0, y0), (x1, y1), (80, 220, 80), 1)
        else:
            cv2.circle(canvas, (x0, y0), 4, (60, 60, 230), -1)
    n_ok = int(st.sum())
    cv2.putText(canvas, f"{cam} frame {i} -> {i+gap}: {n_ok}/{len(st)} matched",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, nargs="+", default=[200, 900, 1600, 2200])
    ap.add_argument("--gap", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for cam in ["cam0", "cam1"]:
        for i in args.frames:
            canvas = match_pair(args.dataset, i, args.gap, cam)
            if canvas is None:
                continue
            f = args.out / f"matches_{cam}_f{i}_gap{args.gap}.png"
            cv2.imwrite(str(f), canvas)
            print(f)


if __name__ == "__main__":
    main()
