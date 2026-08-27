#!/usr/bin/env python
"""Visualize a COLMAP sparse model in Rerun (+ export PLY).

Logs the 3D point cloud (with COLMAP RGB) and per-lens camera trajectories.
Usage:
  python scripts/viz_colmap_rerun.py <sparse_model_dir> <out.rrd> [--scale M_PER_UNIT] [--ply out.ply]
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import rerun as rr


def read_points3d(p: Path):
    pts, cols = [], []
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<Q", f.read(8))  # id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            struct.unpack("<d", f.read(8))  # error
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tl)
            pts.append(xyz)
            cols.append(rgb)
    return np.array(pts), np.array(cols, dtype=np.uint8)


def read_images(p: Path):
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
            imgs[name.decode()] = -R.T @ t  # camera center in world
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", type=float, default=1.0, help="meters per model unit")
    ap.add_argument("--ply", type=Path, default=None)
    args = ap.parse_args()

    pts, cols = read_points3d(args.model / "points3D.bin")
    centers = read_images(args.model / "images.bin")
    s = args.scale
    print(f"{len(pts)} points, {len(centers)} cameras, scale {s} m/unit")

    rr.init("colmap_cloud", spawn=False)
    rr.save(str(args.out))
    rr.log("cloud", rr.Points3D(pts * s, colors=cols, radii=0.008), static=True)
    for cam, color in [("cam0", [42, 120, 214]), ("cam1", [235, 104, 52])]:
        traj = np.array([c for n, c in sorted(centers.items()) if n.startswith(cam)]) * s
        if len(traj):
            rr.log(f"traj/{cam}", rr.LineStrips3D([traj], colors=[color]), static=True)
    print(f"saved {args.out}")

    if args.ply:
        with open(args.ply, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(pts * s, cols):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        print(f"saved {args.ply}")


if __name__ == "__main__":
    main()
