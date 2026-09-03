#!/usr/bin/env python
"""Spherical line features on a fisheye — extraction + health measurement.

This is the winning entry's formulation (ACDC-VSLAM, #1):
    "ELSED segments, endpoints lifted to bearings, represented by the normal of
     the associated great circle. Filtered by angular length; tracked with
     gating on direction, normal alignment, length consistency."

WHY LINES: they are the clearest top-2-only signal in the whole field (only #1
and #2 use them). And they survive exactly what killed us -- low texture, motion
blur, corridors and doorways -- because a line needs only one gradient
direction, not a locally-unique patch.

THE GEOMETRY (why lines are represented as a normal, not two endpoints):
A straight line in the world projects, on a central camera, to a GREAT CIRCLE on
the unit sphere. That circle is fully described by its plane normal

    n = b1 x b2 / |b1 x b2|          b1, b2 = endpoint bearings

and any bearing b lying on the line satisfies  n . b = 0.  That residual is
invariant to sliding along the line, which is exactly the aperture problem: a
line carries ONE constraint, not two, and this formulation stops us pretending
otherwise.

ANGULAR length, not pixel length: on a fisheye the rim is compressed, so a
segment that is short in pixels can span a large angle. Filtering in pixels
throws away the widest-baseline lines -- the most valuable ones.

Usage:
  line_health.py --dataset <ds> --imucam <kalibr.yaml> [--frames a:b] [--out csv]
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def load_cam(path, cam=0):
    blk = re.split(r"\ncam\d+:", Path(path).read_text())[1 + cam]
    fx, fy, cx, cy = [float(x) for x in
                      re.search(r"intrinsics:\s*\[(.*?)\]", blk).group(1).split(",")]
    d = [float(x) for x in
         re.search(r"distortion_coeffs:\s*\[(.*?)\]", blk).group(1).split(",")]
    w, h = [int(float(x)) for x in
            re.search(r"resolution:\s*\[(.*?)\]", blk).group(1).split(",")]
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, d=d, w=w, h=h)


def unproject_kb4(u, v, cam, iters=8):
    """pixel -> unit bearing, KB4 (Kannala-Brandt). Newton on theta_d(theta)."""
    mx = (u - cam["cx"]) / cam["fx"]
    my = (v - cam["cy"]) / cam["fy"]
    ru = np.hypot(mx, my)
    k1, k2, k3, k4 = cam["d"]
    th = ru.copy()                      # theta_d ~ theta for small angles
    for _ in range(iters):
        th2 = th * th
        f = th * (1 + k1*th2 + k2*th2**2 + k3*th2**3 + k4*th2**4) - ru
        df = 1 + 3*k1*th2 + 5*k2*th2**2 + 7*k3*th2**3 + 9*k4*th2**4
        th = th - f / np.maximum(df, 1e-9)
    th = np.clip(th, 0, np.pi * 0.55)
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    return b / np.linalg.norm(b, axis=-1, keepdims=True)


def segments_to_planes(segs, cam, min_ang_deg):
    """(N,4) pixel segments -> (normals, angular_lengths_deg), angular-filtered."""
    if segs is None or len(segs) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    s = np.asarray(segs).reshape(-1, 4)
    b1 = unproject_kb4(s[:, 0], s[:, 1], cam)
    b2 = unproject_kb4(s[:, 2], s[:, 3], cam)
    n = np.cross(b1, b2)
    ln = np.linalg.norm(n, axis=1)
    ang = np.degrees(np.arcsin(np.clip(ln, 0, 1)))     # angular length of the arc
    ok = (ln > 1e-6) & (ang >= min_ang_deg)
    return n[ok] / ln[ok, None], ang[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--imucam", required=True)
    ap.add_argument("--frames", default=None, help="first:last frame id")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-ang", type=float, default=1.5,
                    help="minimum ANGULAR length [deg]")
    ap.add_argument("--out", default=None, help="write per-frame counts here")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ds = Path(a.dataset)
    cams = [load_cam(a.imucam, 0), load_cam(a.imucam, 1)]
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int)
    ft = np.atleast_1d(fr["t"])
    if a.frames:
        lo, hi = (int(x) for x in a.frames.split(":"))
        m = (fid >= lo) & (fid <= hi)
        fid, ft = fid[m], ft[m]
    fid, ft = fid[::a.stride], ft[::a.stride]

    lsd = cv2.createLineSegmentDetector()
    rows = []
    for k, t in zip(fid, ft):
        cnt = []
        for c in (0, 1):
            img = cv2.imread(str(ds / f"cam{c}" / f"{k:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                cnt.append(0); continue
            segs = lsd.detect(img)[0]
            n, ang = segments_to_planes(segs, cams[c], a.min_ang)
            cnt.append(len(n))
        rows.append((t, cnt[0], cnt[1]))
    r = np.array(rows)
    lab = a.label or ds.name
    print(f"{lab:22s} frames {len(r):5d}  "
          f"lines/frame  cam0 med {np.median(r[:,1]):6.0f} min {r[:,1].min():4.0f}  "
          f"cam1 med {np.median(r[:,2]):6.0f} min {r[:,2].min():4.0f}  "
          f"frames with <5 lines total: {int(((r[:,1]+r[:,2])<5).sum())}")
    if a.out:
        np.savetxt(a.out, r, delimiter=",", header="t,lines_cam0,lines_cam1",
                   comments="")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
