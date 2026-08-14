#!/usr/bin/env python
"""SLAM-style temporal replay of a COLMAP reconstruction in Rerun.

Standard layout: 3D map (cloud + growing trajectory + rig camera frustums) +
reprojection-error plot on the left; both camera feeds with COLMAP's registered
keypoints overlaid on the right. Everything on the video timeline.

Usage:
  viz_gt_replay.py <sparse_model_dir> <dataset_dir> <out.rrd> --scale M_PER_UNIT
"""
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from viz_colmap_rerun import read_points3d

IMG_SCALE = 0.2


def read_model(model: Path):
    cams = {}
    with open(model / "cameras.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, m, w, h = struct.unpack("<iiQQ", f.read(24))
            cams[cid] = (w, h, np.array(struct.unpack("<8d", f.read(64))))
    imgs = {}
    with open(model / "images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            pts = np.frombuffer(f.read(24 * npts), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            imgs[name.decode()] = (q, t, cid, pts)
    p3d = {}
    with open(model / "points3D.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3 + 8)
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tl)
            p3d[pid] = np.array(xyz)
    return cams, imgs, p3d


def qtoR(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def kb4_project(X, params):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(r, z)
    d = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    inv_r = np.where(r > 1e-9, 1.0 / r, 0.0)
    return np.stack([fx * d * x * inv_r + cx, fy * d * y * inv_r + cy], 1)


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
            rrb.Vertical(
                rrb.Spatial3DView(origin="/world", name="map", contents="/world/**"),
                rrb.TimeSeriesView(origin="/plots/reproj_rms_px", name="reprojection rms [px]"),
                row_shares=[4, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/cam0", name="cam0", contents="/cam0/**"),
                rrb.Spatial2DView(origin="/cam1", name="cam1", contents="/cam1/**"),
            ),
            column_shares=[2, 1],
        )
    )
    rr.init("colmap_replay", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)

    pts, cols = read_points3d(args.model / "points3D.bin")
    rr.log("world/cloud", rr.Points3D(pts * args.scale, colors=cols, radii=0.01), static=True)

    cams, imgs, p3d = read_model(args.model)
    by_frame = {}
    for name, (q, t, cid, pts2d) in imgs.items():
        cam, fn = name.split("/")
        by_frame.setdefault(int(fn.split(".")[0]), {})[cam] = (q, t, cid, pts2d)

    path = []
    for fr in sorted(by_frame):
        tsec = (fr - 1) / args.fps
        set_t(tsec)
        rms_all, n_all = 0.0, 0
        for cam in ("cam0", "cam1"):
            if cam not in by_frame[fr]:
                continue
            q, t, cid, pts2d = by_frame[fr][cam]
            R = qtoR(q)
            C = -R.T @ t
            if cam == "cam0":
                path.append(C * args.scale)
                rr.log("world/gt_traj", rr.LineStrips3D([np.array(path)], colors=[[27, 175, 122]]))
            # rig frustum: camera pose + pinhole (nominal 90deg fov for display)
            w, h, params = cams[cid]
            rr.log(f"world/rig/{cam}", rr.Transform3D(translation=C * args.scale, mat3x3=R.T))
            rr.log(f"world/rig/{cam}", rr.Pinhole(resolution=[w * IMG_SCALE, h * IMG_SCALE],
                                                  focal_length=float(w * IMG_SCALE / 2), image_plane_distance=0.25))
            # image + registered keypoints + reprojection residuals
            img = cv2.imread(str(args.dataset / cam / f"{fr:06d}.jpg"), 0)
            if img is not None:
                img = cv2.resize(img, None, fx=IMG_SCALE, fy=IMG_SCALE)
                rr.log(f"{cam}/image", rr.Image(img).compress(jpeg_quality=70))
            valid = pts2d[pts2d["id"] >= 0]
            if len(valid):
                uv = np.stack([valid["x"], valid["y"]], 1)
                rr.log(f"{cam}/image/keypoints", rr.Points2D(uv * IMG_SCALE, colors=[[27, 175, 122]], radii=1.5))
                X = np.stack([p3d[i] for i in valid["id"]])
                Xc = (R @ X.T).T + t
                uv_proj = kb4_project(Xc, params)
                res = np.linalg.norm(uv_proj - uv, axis=1)
                rms_all += float((res**2).sum())
                n_all += len(res)
        if n_all:
            rr.log("plots/reproj_rms_px", rr.Scalars(float(np.sqrt(rms_all / n_all))))
    print(f"saved {args.out} ({len(path)} poses)")


if __name__ == "__main__":
    main()
