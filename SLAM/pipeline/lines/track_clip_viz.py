#!/usr/bin/env python
"""Render track_clip output: sampled frames, one persistent colour + id per
TRACK. An edge keeping its colour across the strip = identity survived.
Tracks alive >= --min-life frames get colour+label; the one-frame noise pool
is thin grey. Rotated 180 (sensor upside down).

usage: track_clip_viz.py <clip.csv> <frames_dir> [--n 6] [--min-life 10] [--out DIR]
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("frames_dir", type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--min-life", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.clip.parent

    d = np.genfromtxt(args.clip, delimiter=",", names=True)
    frames = np.unique(d["frame"]).astype(int)
    life = Counter(d["trackId"].astype(int))
    keep = {t for t, c in life.items() if c >= args.min_life}
    rng = np.random.default_rng(11)
    col = {t: tuple(int(c) for c in rng.integers(70, 255, 3)) for t in keep}

    picks = frames[np.linspace(0, len(frames) - 1, args.n).astype(int)]
    tiles = []
    for f in picks:
        img = cv2.imread(str(args.frames_dir / f"{f:06d}.jpg"))
        img = cv2.rotate(img, cv2.ROTATE_180)
        h, w = img.shape[:2]
        rows = d[d["frame"] == f]
        for r in np.atleast_1d(rows):
            t = int(r["trackId"])
            p1 = (int(w - 1 - r["x1"]), int(h - 1 - r["y1"]))
            p2 = (int(w - 1 - r["x2"]), int(h - 1 - r["y2"]))
            if t in keep:
                cv2.line(img, p1, p2, col[t], 3, cv2.LINE_AA)
                cv2.putText(img, str(t), ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col[t], 2, cv2.LINE_AA)
            else:
                cv2.line(img, p1, p2, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {f}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 255), 4, cv2.LINE_AA)
        tiles.append(img)
    cols = 3
    rows_n = (len(tiles) + cols - 1) // cols
    H, W = tiles[0].shape[:2]
    canvas = np.zeros((rows_n * H, cols * W, 3), np.uint8)
    for k, tile in enumerate(tiles):
        y, x = divmod(k, cols)
        canvas[y*H:(y+1)*H, x*W:(x+1)*W] = tile
    sc = 2400 / canvas.shape[1]
    canvas = cv2.resize(canvas, None, fx=sc, fy=sc)
    p = out / "clip_identity.png"
    cv2.imwrite(str(p), canvas)
    print(f"{p}  ({len(keep)} tracks with life >= {args.min_life} coloured)")


if __name__ == "__main__":
    main()
