#!/usr/bin/env python
"""Extract + validate Insta360 factory calibration embedded in .insv trailers.

The trailer holds two underscore-separated per-unit calibration strings:
  34 fields: polynomial model  (f, cx, cy, euler, t, poly[4]) x 2 lenses
  40 fields: Mei/UCM + radtan  (xi, fx, fy, cx, cy, euler, t, dist[5]) x 2 lenses
Coordinates are on the 6272x3072 dual-photo layout (per-lens half 3136x3072);
2944x2880 video frames are a central crop => subtract (96, 96) from centers
(and 3136 from cx for lens1).

Commands:
  parse    <video.insv> -o calib.json
  validate <video.insv> <frame.jpg> --cam 0|1 -o montage.png
      Renders virtual-pinhole views from the fisheye frame under candidate
      distortion-order interpretations. Straight lines straight = correct one.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CROP = 96          # sensor -> 2944x2880 video central crop
HALF_W = 3136      # per-lens width in the 6272x3072 layout


def find_calib_strings(video: Path) -> dict:
    data = video.read_bytes()[-4 * 1024 * 1024:]
    out = {}
    for m in re.finditer(rb"[-0-9.]+(?:_[-0-9.]+){20,}", data):
        f = m.group().decode().split("_")
        if len(f) == 34:
            out["poly"] = [
                {"f": float(f[1 + 16 * i]), "cx": float(f[2 + 16 * i]), "cy": float(f[3 + 16 * i]),
                 "euler_deg": [float(x) for x in f[4 + 16 * i:7 + 16 * i]],
                 "t_m": [float(x) for x in f[7 + 16 * i:10 + 16 * i]],
                 "poly": [float(x) for x in f[10 + 16 * i:14 + 16 * i]]}
                for i in range(2)]
        elif len(f) == 40:
            out["mei"] = [
                {"xi": float(f[1 + 19 * i]), "fx": float(f[2 + 19 * i]), "fy": float(f[3 + 19 * i]),
                 "cx": float(f[4 + 19 * i]), "cy": float(f[5 + 19 * i]),
                 "euler_deg": [float(x) for x in f[6 + 19 * i:9 + 19 * i]],
                 "t_m": [float(x) for x in f[9 + 19 * i:12 + 19 * i]],
                 "dist": [float(x) for x in f[12 + 19 * i:17 + 19 * i]]}
                for i in range(2)]
    return out


def video_frame_params(mei_lens: dict, cam: int) -> dict:
    """Map sensor-layout Mei params onto a 2944x2880 video frame."""
    p = dict(mei_lens)
    cx = p["cx"] - (HALF_W if cam == 1 else 0) - CROP
    cy = p["cy"] - CROP
    return {**p, "cx": cx, "cy": cy}


def mei_project(rays: np.ndarray, p: dict, order: str) -> np.ndarray:
    """rays (N,3) in lens frame -> pixel coords (N,2). order: dist interpretation."""
    Xs = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    z = Xs[:, 2] + p["xi"]
    x, y = Xs[:, 0] / z, Xs[:, 1] / z
    c1, c2, c3, c4, c5 = p["dist"]
    r2 = x * x + y * y
    if order == "k1k2p1p2":          # c3 unused
        rad = 1 + c1 * r2 + c2 * r2 ** 2
        p1, p2 = c4, c5
    elif order == "k1k2k3p1p2":      # radial cubic
        rad = 1 + c1 * r2 + c2 * r2 ** 2 + c3 * r2 ** 3
        p1, p2 = c4, c5
    else:                            # "none": no distortion reference
        rad, p1, p2 = 1.0, 0.0, 0.0
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([p["fx"] * xd + p["cx"], p["fy"] * yd + p["cy"]], axis=1)


def render_pinhole(img: np.ndarray, p: dict, order: str, yaw_deg: float,
                   fov_deg: float = 90.0, size: int = 640) -> np.ndarray:
    f = size / (2 * np.tan(np.radians(fov_deg) / 2))
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    rays = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones_like(u, float)], -1).reshape(-1, 3)
    yw = np.radians(yaw_deg)
    R = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
    uv = mei_project(rays @ R.T, p, order)
    ui = np.clip(uv[:, 0].round().astype(int), 0, img.shape[1] - 1)
    vi = np.clip(uv[:, 1].round().astype(int), 0, img.shape[0] - 1)
    valid = ((uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) &
             (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0]))
    out = img[vi, ui]
    out[~valid] = 0
    return out.reshape(size, size, 3)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("parse")
    p1.add_argument("video", type=Path)
    p1.add_argument("-o", "--out", type=Path, required=True)
    p2 = sub.add_parser("validate")
    p2.add_argument("video", type=Path)
    p2.add_argument("frame", type=Path)
    p2.add_argument("--cam", type=int, default=0)
    p2.add_argument("-o", "--out", type=Path, required=True)
    p2.add_argument("--rot180-display", action="store_true",
                    help="frame is in sensor orientation (upside down); rotate "
                         "rendered tiles for display only")
    args = ap.parse_args()

    calib = find_calib_strings(args.video)
    if args.cmd == "parse":
        args.out.write_text(json.dumps(calib, indent=2))
        print(f"{list(calib.keys())} -> {args.out}")
        return

    p = video_frame_params(calib["mei"][args.cam], args.cam)
    img = np.asarray(Image.open(args.frame).convert("RGB"))
    orders = ["none", "k1k2p1p2", "k1k2k3p1p2"]
    yaws = [0, 65]
    S, PAD, LBL = 640, 8, 30
    canvas = Image.new("RGB", ((S + PAD) * len(yaws) + PAD, (S + PAD + LBL) * len(orders) + PAD), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    for i, order in enumerate(orders):
        y0 = PAD + i * (S + PAD + LBL)
        d.text((PAD, y0 + 4), f"dist = {order}   (rows: forward view | 65-deg side view; "
                              f"judge: straight lines straight?)", fill=(220, 220, 220))
        for j, yaw in enumerate(yaws):
            tile = render_pinhole(img, p, order, yaw)
            if args.rot180_display:
                tile = tile[::-1, ::-1]
            canvas.paste(Image.fromarray(tile), (PAD + j * (S + PAD), y0 + LBL))
    canvas.save(args.out)
    print(f"cam{args.cam} params on video frame: xi={p['xi']}, fx={p['fx']:.1f}, "
          f"cx={p['cx']:.1f}, cy={p['cy']:.1f}")
    print(f"montage -> {args.out}")


if __name__ == "__main__":
    main()
