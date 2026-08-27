#!/usr/bin/env python3
"""
Visualize one FIORD example: the two raw fisheyes + the MediaSDK equirect,
and a sweep of post-hoc spherical rotations of the equirect (to hunt the
gravity-correct orientation).

Outputs (into outputs/viz/):
  <id>_fisheyes_eq.png    : fisheye1 | fisheye2 | equirect
  <id>_rotation_sweep.png : grid of the equirect under different rotations
  rotated/<id>_<tag>.png  : full-res rotated equirects for the sweep angles
Nothing is opened/viewed here; files are only written.
"""
import os, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FI   = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"


def load_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def euler_R(pitch_deg, yaw_deg, roll_deg):
    """Rotation matrix from pitch(x), yaw(y), roll(z) in degrees. y is up."""
    p, y, r = np.deg2rad([pitch_deg, yaw_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotate_equirect(img, R):
    """Rotate an equirectangular image by 3x3 R (applied to the scene)."""
    H, W = img.shape[:2]
    u = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi          # lon [-pi,pi)
    v = (np.arange(H) + 0.5) / H * np.pi                       # colat [0,pi]
    lon, colat = np.meshgrid(u, v)
    lat = np.pi / 2 - colat
    # output direction (y up)
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    d = np.stack([x, y, z], axis=-1)                          # H,W,3
    d_in = d @ R                                              # = (R^T d^T)^T, samples rotated scene
    xi, yi, zi = d_in[..., 0], d_in[..., 1], d_in[..., 2]
    lon_i = np.arctan2(xi, zi)
    lat_i = np.arcsin(np.clip(yi, -1, 1))
    map_x = ((lon_i + np.pi) / (2 * np.pi) * W).astype(np.float32)
    map_y = ((np.pi / 2 - lat_i) / np.pi * H).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def down(img, maxw):
    h, w = img.shape[:2]
    if w <= maxw:
        return img
    s = maxw / w
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="729")
    ap.add_argument("--eq", default=f"{ROOT}/outputs/mediasdk/single/729_base.png")
    ap.add_argument("--outdir", default=f"{ROOT}/outputs/viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/rotated", exist_ok=True)

    # locate the two fisheyes for this id
    f1 = [f for f in os.listdir(f"{FI}/cam1") if f"_{args.id}_" in f][0]
    f2 = [f for f in os.listdir(f"{FI}/cam2") if f"_{args.id}_" in f][0]
    fish1 = load_rgb(f"{FI}/cam1/{f1}")
    fish2 = load_rgb(f"{FI}/cam2/{f2}")
    eq = load_rgb(args.eq)

    # ---- Fig A: fisheye1 | fisheye2 | equirect ----
    fig, ax = plt.subplots(1, 3, figsize=(22, 6))
    ax[0].imshow(down(fish1, 1000)); ax[0].set_title(f"fisheye1 (cam1) {fish1.shape[1]}x{fish1.shape[0]}")
    ax[1].imshow(down(fish2, 1000)); ax[1].set_title(f"fisheye2 (cam2) {fish2.shape[1]}x{fish2.shape[0]}")
    ax[2].imshow(down(eq, 1600));    ax[2].set_title(f"equirect (MediaSDK optflow) {eq.shape[1]}x{eq.shape[0]}")
    for a in ax: a.axis("off")
    fig.suptitle(f"FIORD MeetingRoom  id={args.id}", fontsize=16)
    fig.tight_layout()
    pA = f"{args.outdir}/{args.id}_fisheyes_eq.png"
    fig.savefig(pA, dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- Fig B: rotation sweep ----
    # (tag, pitch, yaw, roll)
    rots = [
        ("identity",      0,   0,   0),
        ("pitch+90",     90,   0,   0),
        ("pitch-90",    -90,   0,   0),
        ("roll+90",       0,   0,  90),
        ("roll-90",       0,   0, -90),
        ("yaw+90",        0,  90,   0),
        ("yaw+180",       0, 180,   0),
        ("pitch+90_roll90",90,  0,  90),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(24, 7))
    ax = ax.ravel()
    for i, (tag, p, y, r) in enumerate(rots):
        rimg = eq if tag == "identity" else rotate_equirect(eq, euler_R(p, y, r))
        # save full-res rotated (skip identity dup)
        if tag != "identity":
            cv2.imwrite(f"{args.outdir}/rotated/{args.id}_{tag}.png",
                        cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR))
        ax[i].imshow(down(rimg, 900)); ax[i].set_title(tag); ax[i].axis("off")
    fig.suptitle(f"Rotation sweep of equirect  id={args.id}  (p=pitch/x, y=yaw/up, r=roll/z)", fontsize=15)
    fig.tight_layout()
    pB = f"{args.outdir}/{args.id}_rotation_sweep.png"
    fig.savefig(pB, dpi=110, bbox_inches="tight"); plt.close(fig)

    print("WROTE:")
    print(" ", pA)
    print(" ", pB)
    print(" ", f"{args.outdir}/rotated/  (full-res rotated eqs)")


if __name__ == "__main__":
    main()
