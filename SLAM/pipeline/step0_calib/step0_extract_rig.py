#!/usr/bin/env python
"""Extract the rig extrinsics from a Step 0 reconstruction.

cam0/NNNNNN.jpg and cam1/NNNNNN.jpg are the SAME instant, so any model that
registered both gives us a direct measurement of the rig:

    T_cam0_cam1 = T_cam0_world * inv(T_cam1_world)

We report the distribution over all such frames, not just a mean -- a tight
distribution means the rig is observable; a wide one means the cross-camera
links are too weak to trust, and we should say so rather than quote a number.

Ground truth for Hilti: 179.56 deg between optical axes, 4.01 cm baseline.

Usage: step0_extract_rig.py <sparse_dir> [--truth-deg 179.56 --truth-cm 4.01]
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def read_images_bin(path):
    """COLMAP images.bin -> {name: (qvec, tvec)}, world-to-camera."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<i", f.read(4))                      # image_id
            q = np.array(struct.unpack("<4d", f.read(32)))      # w,x,y,z
            t = np.array(struct.unpack("<3d", f.read(24)))
            struct.unpack("<i", f.read(4))                      # camera_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)
            out[name.decode()] = (q, t)
    return out


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sparse", type=Path)
    ap.add_argument("--truth-deg", type=float, default=179.56)
    ap.add_argument("--truth-cm", type=float, default=4.01)
    args = ap.parse_args()

    models = sorted([d for d in args.sparse.iterdir() if (d / "images.bin").exists()],
                    key=lambda d: (d / "images.bin").stat().st_size, reverse=True)
    print(f"{len(models)} model(s) found\n")

    best = None
    for d in models:
        imgs = read_images_bin(d / "images.bin")
        f0 = {n.split("/")[1]: v for n, v in imgs.items() if n.startswith("cam0/")}
        f1 = {n.split("/")[1]: v for n, v in imgs.items() if n.startswith("cam1/")}
        shared = sorted(set(f0) & set(f1))
        if not shared:
            print(f"model {d.name}: {len(imgs):4d} imgs, cam0={len(f0)} cam1={len(f1)}, "
                  f"shared frames=0  (no rig observation)")
            continue

        angs, bases = [], []
        for k in shared:
            R0, t0 = qvec2rot(f0[k][0]), f0[k][1]
            R1, t1 = qvec2rot(f1[k][0]), f1[k][1]
            # world->cam convention: X_c = R X_w + t  =>  T_c0_c1 = T_c0_w * inv(T_c1_w)
            R = R0 @ R1.T
            t = t0 - R @ t1
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            angs.append(ang)
            bases.append(np.linalg.norm(t))
        angs, bases = np.array(angs), np.array(bases)
        print(f"model {d.name}: {len(imgs):4d} imgs, shared frames={len(shared):3d}  "
              f"angle med {np.median(angs):7.2f} deg (IQR {np.percentile(angs,75)-np.percentile(angs,25):5.2f})  "
              f"baseline med {np.median(bases):.4f} (arb. scale)")
        if best is None or len(shared) > best[0]:
            best = (len(shared), d, angs, bases)

    if best is None:
        print("\nNo model registered both cameras -> rig NOT observable from this run.")
        return

    n, d, angs, bases = best
    print(f"\n=== best: model {d.name}, {n} shared frames ===")
    print(f"inter-camera ANGLE   median {np.median(angs):.3f} deg   "
          f"[p25 {np.percentile(angs,25):.3f}, p75 {np.percentile(angs,75):.3f}]")
    print(f"   ground truth {args.truth_deg} deg  ->  error {np.median(angs)-args.truth_deg:+.3f} deg")
    print(f"   (180.0 assumption would be off by {args.truth_deg-180.0:+.3f} deg)")
    # SfM scale is arbitrary; report the baseline ratio spread, which is scale-free
    rel = bases / np.median(bases)
    print(f"\nBASELINE is in arbitrary SfM scale: median {np.median(bases):.4f}, "
          f"relative spread p25-p75 = {np.percentile(rel,25):.3f}-{np.percentile(rel,75):.3f}")
    print("   -> metric baseline needs a scale anchor (IMU, or a known distance).")


if __name__ == "__main__":
    main()
