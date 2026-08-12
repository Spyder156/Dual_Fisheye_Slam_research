#!/usr/bin/env python3
"""
Visualize COLMAP sparse reconstructions (arm A = MediaSDK equirect,
arm B = FIORD fisheye reference): point cloud (top + side) and camera
trajectory, one figure per arm + a stats bar. Reads COLMAP TXT exports.
Only writes files; nothing is opened/viewed here.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
ARMS = {
    "A_MediaSDK_equirect": f"{ROOT}/outputs/benchmark/armA_mediasdk_equirect/sparse/0",
    "B_FIORD_fisheye":      f"{ROOT}/outputs/viz/clouds/B_txt",
}
OUT = f"{ROOT}/outputs/viz/recon"
os.makedirs(OUT, exist_ok=True)


def qvec2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def read_points(path):
    xyz, rgb = [], []
    with open(f"{path}/points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.array(xyz), np.array(rgb) / 255.0


def read_cams(path):
    C = []
    with open(f"{path}/images.txt") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):  # pose line, then points line
        p = lines[i].split()
        q = np.array([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        t = np.array([float(p[5]), float(p[6]), float(p[7])])
        R = qvec2R(q)
        C.append(-R.T @ t)  # camera center in world
    return np.array(C)


def robust_lims(P, pad=0.05, lo=1, hi=99):
    mn = np.percentile(P, lo, axis=0)
    mx = np.percentile(P, hi, axis=0)
    d = (mx - mn) * pad
    return mn - d, mx + d


for name, path in ARMS.items():
    xyz, rgb = read_points(path)
    C = read_cams(path)
    # clip outliers for display
    mn, mx = robust_lims(xyz)
    m = np.all((xyz >= mn) & (xyz <= mx), axis=1)
    P, col = xyz[m], rgb[m]

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    # top view (X-Z)  and side view (X-Y)
    for k, (i, j, lbl) in enumerate([(0, 2, "top  (X-Z)"), (0, 1, "side (X-Y)")]):
        ax[k].scatter(P[:, i], P[:, j], s=0.4, c=col, alpha=0.5, linewidths=0)
        ax[k].plot(C[:, i], C[:, j], "-", color="red", lw=1.0, alpha=0.7)
        ax[k].scatter(C[:, i], C[:, j], s=14, c="red", label=f"{len(C)} cameras")
        ax[k].set_title(f"{lbl}   pts={len(xyz)}")
        ax[k].set_aspect("equal", "box")
        ax[k].legend(loc="upper right")
    fig.suptitle(f"{name}   ({len(C)} cams, {len(xyz)} points)", fontsize=15)
    fig.tight_layout()
    outp = f"{OUT}/{name}.png"
    fig.savefig(outp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", outp, f"(cams={len(C)}, pts={len(xyz)})")
