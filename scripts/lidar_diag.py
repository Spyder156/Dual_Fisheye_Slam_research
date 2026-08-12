#!/usr/bin/env python3
"""
Lidar diagnostic (step 1 before C2C):
  - load lidar (.e57 all scans, or the small .ply)
  - load COLMAP A/B point clouds
  - print point counts + bbox extents (to see scale mismatch)
  - is arm B already ~aligned to lidar? (top-view overlay)
Writes a downsampled lidar .ply for reuse. Only writes files.
"""
import os, numpy as np

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FI = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom"
OUT = f"{ROOT}/outputs/viz/lidar"
os.makedirs(OUT, exist_ok=True)


def load_e57(path, max_pts=3_000_000):
    import pye57
    e = pye57.E57(path)
    n = e.scan_count
    allpts = []
    for i in range(n):
        d = e.read_scan(i, ignore_missing_fields=True, colors=False, intensity=False)
        p = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], axis=1)
        allpts.append(p)
    P = np.concatenate(allpts, axis=0)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) > max_pts:
        idx = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
        P = P[idx]
    return P, n


def read_colmap_pts(path):
    xyz = []
    with open(f"{path}/points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(xyz)


def extent(P):
    lo = np.percentile(P, 1, axis=0)
    hi = np.percentile(P, 99, axis=0)
    return hi - lo


print("=== LIDAR (.e57) ===")
L, nscan = load_e57(f"{FI}/meetingroom.e57")
print(f"scans={nscan}  points(sampled)={len(L)}")
print("extent (99-1 pct) XYZ:", np.round(extent(L), 3), "  (meters if metric)")

import open3d as o3d
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(L)
pc = pc.voxel_down_sample(0.02)
o3d.io.write_point_cloud(f"{OUT}/lidar_ds.ply", pc)
print(f"saved downsampled lidar: {OUT}/lidar_ds.ply  ({len(pc.points)} pts @2cm voxel)")

A = read_colmap_pts(f"{ROOT}/outputs/benchmark/armA_mediasdk_equirect/sparse/0")
B = read_colmap_pts(f"{ROOT}/outputs/viz/clouds/B_txt")
print("\n=== COLMAP clouds ===")
print("A extent:", np.round(extent(A), 3), " pts:", len(A))
print("B extent:", np.round(extent(B), 3), " pts:", len(B))
print("\nscale ratio lidar/B (per-axis, sorted):",
      np.round(np.sort(extent(L)) / np.sort(extent(B)), 3))
print("scale ratio lidar/A (per-axis, sorted):",
      np.round(np.sort(extent(L)) / np.sort(extent(A)), 3))

# top-view overlay: is B already aligned to lidar?
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(20, 9))
ax[0].scatter(L[::10, 0], L[::10, 1], s=0.2, c="gray", alpha=0.4, label="lidar")
ax[0].scatter(B[:, 0], B[:, 1], s=0.5, c="red", alpha=0.5, label="arm B (raw frame)")
ax[0].set_title("lidar vs B  (X-Y, raw frames — expect misaligned)"); ax[0].legend(); ax[0].set_aspect("equal", "box")
ax[1].scatter(L[::10, 0], L[::10, 2], s=0.2, c="gray", alpha=0.4)
ax[1].set_title("lidar only (X-Z)"); ax[1].set_aspect("equal", "box")
fig.tight_layout(); fig.savefig(f"{OUT}/diag_overlay.png", dpi=100); plt.close(fig)
print(f"\nsaved: {OUT}/diag_overlay.png")
