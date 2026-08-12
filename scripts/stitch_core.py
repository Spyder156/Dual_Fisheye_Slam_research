#!/usr/bin/env python3
"""
Arm C — deterministic ray-consistent stitch core (step 1: single lens).

For each equirect pixel: direction -> (rotate into lens frame) -> Kannala-Brandt
(OpenCV-Fisheye) projection -> sample fisheye. Emits equirect + honest validity
mask (invalid where the ray leaves the lens FoV or the image). Fixed function,
no content adaptation.

This step: project ONE lens (cam1) into an equirect hemisphere + mask, and
visualize fisheye | our equirect | mask. Nothing viewed here; only writes.
"""
import os, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
OUT = f"{ROOT}/outputs/armC"; os.makedirs(OUT, exist_ok=True)

# FIORD cam1 OPENCV_FISHEYE intrinsics
CAM1 = dict(fx=998.71442825746783, fy=1006.4807034467711, cx=1612.0, cy=1617.0,
            k=[0.034358190499964164, -0.01780110435914365,
               -0.00079837513258086007, 0.00036177684527421565],
            w=3264, h=3264)
FOV_DEG = 200.0                      # per-lens FoV -> theta_max = 100 deg


def kb_project(X, Y, Z, cam):
    """Kannala-Brandt (OpenCV fisheye) forward projection. Returns u,v,theta."""
    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(r, Z)
    k1, k2, k3, k4 = cam["k"]
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(r > 1e-9, theta_d / np.where(r > 1e-9, r, 1), 0.0)
    xp, yp = X * scale, Y * scale
    u = cam["fx"] * xp + cam["cx"]
    v = cam["fy"] * yp + cam["cy"]
    return u, v, theta


def equirect_dirs(W, H):
    """direction per equirect pixel; center col/row = +Z (lens forward). y-down."""
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi      # [-pi,pi)
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi       # [pi/2,-pi/2]
    lon, lat = np.meshgrid(lon, lat)
    X = np.cos(lat) * np.sin(lon)
    Y = -np.sin(lat)
    Z = np.cos(lat) * np.cos(lon)
    return X, Y, Z


def render_lens(fisheye_bgr, cam, W=2048, H=1024, R=None):
    X, Y, Z = equirect_dirs(W, H)
    d = np.stack([X, Y, Z], -1)
    if R is not None:
        d = d @ R.T
    X, Y, Z = d[..., 0], d[..., 1], d[..., 2]
    u, v, theta = kb_project(X, Y, Z, cam)
    theta_max = np.deg2rad(FOV_DEG / 2)
    valid = (theta <= theta_max) & (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
    mapx = np.where(valid, u, -1).astype(np.float32)
    mapy = np.where(valid, v, -1).astype(np.float32)
    eq = cv2.remap(fisheye_bgr, mapx, mapy, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    eq[~valid] = 0
    return eq, valid.astype(np.uint8) * 255


def main():
    fid = "729"
    FI = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages/cam1"
    fname = [f for f in os.listdir(FI) if f"_{fid}_" in f][0]
    fisheye = cv2.imread(f"{FI}/{fname}")

    eq, mask = render_lens(fisheye, CAM1)
    cv2.imwrite(f"{OUT}/cam1_{fid}_equirect.png", eq)
    cv2.imwrite(f"{OUT}/cam1_{fid}_mask.png", mask)

    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    ax[0].imshow(rgb(cv2.resize(fisheye, (1000, 1000)))); ax[0].set_title("input fisheye cam1 (3264²)")
    ax[1].imshow(rgb(eq)); ax[1].set_title("our single-lens equirect (KB reprojection)")
    ax[2].imshow(mask, cmap="gray"); ax[2].set_title(f"validity mask (valid={ (mask>0).mean()*100:.1f}%)")
    for a in ax: a.axis("off")
    fig.suptitle(f"Arm C core — single-lens reprojection  cam1 frame {fid}", fontsize=15)
    fig.tight_layout(); fig.savefig(f"{OUT}/single_lens_{fid}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("WROTE:")
    print(f"  {OUT}/single_lens_{fid}.png   (fisheye | equirect | mask)")
    print(f"  {OUT}/cam1_{fid}_equirect.png , cam1_{fid}_mask.png")


if __name__ == "__main__":
    main()
