#!/usr/bin/env python
"""SLAM-style temporal replay of a COLMAP reconstruction in Rerun.

Plays the camera frames on the video timeline while the (metric-scaled) COLMAP
trajectory grows in sync through the static point cloud — so pose errors can be
localized against what the camera actually saw.

Usage:
  viz_gt_replay.py <sparse_model_dir> <dataset_dir> <out.rrd> --scale M_PER_UNIT
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from viz_colmap_rerun import read_points3d
import struct


def read_images_full(p: Path):
    imgs = {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
            w, x, y, z = q
            R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                          [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                          [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            imgs[name.decode()] = -R.T @ t
    return imgs


def set_t(t):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("t", t)
    else:
        rr.set_time("t", duration=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--fps", type=float, default=30000 / 1001)
    args = ap.parse_args()

    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="map", contents="/world/**"),
            rrb.Spatial2DView(origin="/cam0", name="camera", contents="/cam0/**"),
            column_shares=[2, 1],
        )
    )
    rr.init("colmap_replay", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)

    pts, cols = read_points3d(args.model / "points3D.bin")
    rr.log("world/cloud", rr.Points3D(pts * args.scale, colors=cols, radii=0.01), static=True)

    centers = read_images_full(args.model / "images.bin")
    frames = sorted((int(n.split("/")[1].split(".")[0]), n) for n in centers if n.startswith("cam0/"))
    path = []
    for fr, name in frames:
        t = (fr - 1) / args.fps
        set_t(t)
        p = centers[name] * args.scale
        path.append(p)
        rr.log("world/gt_traj", rr.LineStrips3D([np.array(path)], colors=[[27, 175, 122]]))
        rr.log("world/pose", rr.Points3D([p], colors=[[235, 104, 52]], radii=0.06))
        img = cv2.imread(str(args.dataset / "cam0" / f"{fr:06d}.jpg"), 0)
        if img is not None:
            img = cv2.resize(img, None, fx=0.2, fy=0.2)
            rr.log("cam0/image", rr.Image(img).compress(jpeg_quality=70))
    print(f"saved {args.out} ({len(path)} poses)")


if __name__ == "__main__":
    main()
