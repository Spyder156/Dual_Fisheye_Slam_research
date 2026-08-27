#!/usr/bin/env python3
"""
Room-structure registration B(FIORD fisheye)->lidar:
  1. RANSAC floor plane in each -> align gravity (normal->normal)
  2. grid-search scale x yaw (about gravity) by median C2C on a downsample
  3. centroid init translation -> multiscale point-to-plane ICP
Then compose A->lidar via the trusted A<->B sim3 (pose_compare) as a cross-check.
Reports median C2C + fraction within 10cm (a real trust signal for a good align).
Outputs -> outputs/viz/c2c_pyaw/
"""
import os, copy, re, numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FI = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom"
OUT = f"{ROOT}/outputs/viz/c2c_pyaw"; os.makedirs(OUT, exist_ok=True)


def load_lidar(v=0.03):
    import pye57
    e = pye57.E57(f"{FI}/meetingroom.e57")
    d = e.read_scan(0, ignore_missing_fields=True, colors=False, intensity=False)
    P = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
    P = P[np.isfinite(P).all(1)]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    return pc.voxel_down_sample(v)


def load_colmap(path):
    xyz = []
    for l in open(f"{path}/points3D.txt"):
        if l.startswith("#") or not l.strip():
            continue
        p = l.split(); xyz.append(list(map(float, p[1:4])))
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.array(xyz))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pc


def floor_normal(pc, dist=0.03):
    # largest plane; take its normal
    model, inl = pc.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=2000)
    n = np.array(model[:3]); n = n / np.linalg.norm(n)
    return n, inl


def R_align(a, b):
    """rotation bringing unit vec a -> unit vec b"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def Rz(a, axis):
    # rotation by angle a about unit axis
    x, y, z = axis; ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca+x*x*(1-ca), x*y*(1-ca)-z*sa, x*z*(1-ca)+y*sa],
        [y*x*(1-ca)+z*sa, ca+y*y*(1-ca), y*z*(1-ca)-x*sa],
        [z*x*(1-ca)-y*sa, z*y*(1-ca)+x*sa, ca+z*z*(1-ca)]])


def median_c2c(src_pts, tree):
    d, _ = tree.query(src_pts, k=1)
    return np.median(d), (d < 0.10).mean()


def main():
    lidar = load_lidar(0.03)
    lidar.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    Lp = np.asarray(lidar.points)
    nL, _ = floor_normal(lidar)
    tree = cKDTree(Lp)

    B = load_colmap(f"{ROOT}/outputs/viz/clouds/B_txt")
    nB, _ = floor_normal(B)
    Bp = np.asarray(B.points)

    # gravity align
    Rg = R_align(nB, nL)
    Bg = (Rg @ Bp.T).T
    up = nL

    # grid search scale x yaw (coarse downsample for speed)
    idx = np.random.default_rng(0).choice(len(Bg), min(4000, len(Bg)), replace=False)
    Bs = Bg[idx]
    cL = np.median(Lp, 0)
    best = None
    for k in np.linspace(0.25, 0.8, 12):          # scale range (v2 sweep peaked ~0.36-0.44)
        for deg in range(0, 360, 6):               # yaw
            R = Rz(np.deg2rad(deg), up)
            P = (R @ (Bs * k).T).T
            P = P - np.median(P, 0) + cL           # centroid init
            med, frac = median_c2c(P, tree)
            if best is None or med < best[0]:
                best = (med, frac, k, deg)
    med0, frac0, s, deg = best
    print(f"grid best: scale={s:.3f} yaw={deg} median_c2c={med0*100:.1f}cm frac<10cm={frac0:.2f}")

    # build full init transform, ICP refine
    R = Rz(np.deg2rad(deg), up) @ Rg
    Bfull = (R @ (Bp * s).T).T
    Bfull = Bfull - np.median(Bfull, 0) + cL
    bpc = o3d.geometry.PointCloud(); bpc.points = o3d.utility.Vector3dVector(Bfull)
    T = np.eye(4)
    for vv, th in [(0.05, 0.3), (0.03, 0.15), (0.02, 0.08)]:
        ds = bpc.voxel_down_sample(vv)
        ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vv*2, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(
            ds, lidar, th, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T = reg.transformation
    bpc.transform(T)
    Bfin = np.asarray(bpc.points)
    dB, _ = tree.query(Bfin, k=1)
    print(f"B AFTER ICP: median_c2c={np.median(dB)*100:.1f}cm frac<10cm={(dB<0.1).mean():.2f} "
          f"fitness={reg.fitness:.2f} rmse={reg.inlier_rmse*100:.1f}cm")

    o3d.io.write_point_cloud(f"{OUT}/B_aligned.ply", bpc)

    # heatmap + hist
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sc = ax[0].scatter(Bfin[:, 0], Bfin[:, 1], c=np.clip(dB, 0, 0.2), s=1.5, cmap="turbo")
    ax[0].scatter(Lp[::20, 0], Lp[::20, 1], s=0.1, c="lightgray", alpha=0.3, zorder=0)
    ax[0].set_aspect("equal", "box"); plt.colorbar(sc, ax=ax[0], label="dist (m)")
    ax[0].set_title(f"B vs lidar  median={np.median(dB)*100:.1f}cm frac<10cm={(dB<0.1).mean():.2f}")
    ax[1].hist(np.clip(dB, 0, 0.3), bins=100, color="steelblue")
    ax[1].axvline(np.median(dB), c="r"); ax[1].set_xlabel("C2C dist (m)")
    fig.tight_layout(); fig.savefig(f"{OUT}/B_result.png", dpi=110); plt.close(fig)

    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(f"B: scale={s:.3f} yaw={deg} median_c2c={np.median(dB)*100:.1f}cm "
                f"frac<10cm={(dB<0.1).mean():.2f} icp_fitness={reg.fitness:.2f}\n")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
