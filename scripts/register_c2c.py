#!/usr/bin/env python3
"""
Register each COLMAP sparse cloud (arm A = MediaSDK equirect, arm B = FIORD
fisheye) to the metric lidar, then measure cloud-to-cloud distance.

Alignment: robust metric pre-scale -> PCA-axis init (multi-hypothesis) ->
scaled point-to-point ICP, keep best by fitness. Reports fitness + inlier
RMSE so alignment trust is visible before believing C2C numbers.

Outputs (outputs/viz/c2c/):
  <arm>_aligned.ply         COLMAP cloud in lidar frame
  <arm>_heatmap.png         points colored by distance-to-lidar (m)
  <arm>_overlay.png         aligned overlay on lidar (top view)
  c2c_hist.png              A vs B distance histograms
  c2c_summary.txt           the numbers
Only writes files.
"""
import os, itertools, numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FI = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom"
OUT = f"{ROOT}/outputs/viz/c2c"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def load_lidar():
    import pye57
    e = pye57.E57(f"{FI}/meetingroom.e57")
    d = e.read_scan(0, ignore_missing_fields=True, colors=False, intensity=False)
    P = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
    P = P[np.isfinite(P).all(1)]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    pc = pc.voxel_down_sample(0.02)
    return pc


def read_colmap(path):
    xyz = []
    with open(f"{path}/points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.array(xyz))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pc


def robust_radius(P):
    c = np.median(P, 0)
    return np.median(np.linalg.norm(P - c, axis=1))


def pca_axes(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    return c, Vt  # rows = principal axes


def register(colmap, lidar):
    Cp = np.asarray(colmap.points); Lp = np.asarray(lidar.points)
    # robust metric pre-scale
    s = robust_radius(Lp) / robust_radius(Cp)
    Cs = (np.asarray(colmap.points) - Cp.mean(0)) * s
    cc = o3d.geometry.PointCloud(); cc.points = o3d.utility.Vector3dVector(Cs)
    lc, lVt = pca_axes(Lp); Lctr = Lp.mean(0)
    _, cVt = pca_axes(Cs)
    lidar.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

    best = None
    # multi-hypothesis: sign flips + a couple axis swaps of PCA frame
    dets = []
    for signs in itertools.product([1, -1], repeat=3):
        for perm in [(0, 1, 2), (1, 0, 2), (0, 2, 1)]:
            Rc = (cVt[list(perm)] * np.array(signs)[:, None])
            if np.linalg.det(Rc) < 0:  # keep proper rotations
                continue
            # map COLMAP-PCA -> lidar-PCA:  R = lVt^T @ Rc
            R = lVt.T @ Rc
            T = np.eye(4); T[:3, :3] = R
            T[:3, 3] = Lctr - R @ Cs.mean(0)
            reg = o3d.pipelines.registration.registration_icp(
                cc, lidar, 0.3, T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            if best is None or reg.fitness > best[0].fitness:
                best = (reg, s)
    return best  # (RegistrationResult on pre-scaled cloud, prescale)


def c2c(colmap_raw, reg, prescale, lidar):
    Cp = np.asarray(colmap_raw.points)
    Cs = (Cp - Cp.mean(0)) * prescale
    cc = o3d.geometry.PointCloud(); cc.points = o3d.utility.Vector3dVector(Cs)
    cc.transform(reg.transformation)
    A = np.asarray(cc.points); L = np.asarray(lidar.points)
    d, _ = cKDTree(L).query(A, k=1)
    return cc, d


def main():
    lidar = load_lidar()
    L = np.asarray(lidar.points)
    arms = {
        "A_MediaSDK": read_colmap(f"{ROOT}/outputs/benchmark/armA_mediasdk_equirect/sparse/0"),
        "B_FIORD":    read_colmap(f"{ROOT}/outputs/viz/clouds/B_txt"),
    }
    summ = []
    dists = {}
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for k, (name, cm) in enumerate(arms.items()):
        (reg, s) = register(cm, lidar)
        cc, d = c2c(cm, reg, s, lidar)
        dists[name] = d
        stats = dict(fitness=reg.fitness, inlier_rmse=reg.inlier_rmse,
                     mean=d.mean(), median=np.median(d),
                     rms=np.sqrt((d**2).mean()), p90=np.percentile(d, 90))
        summ.append((name, stats))
        # save aligned ply
        o3d.io.write_point_cloud(f"{OUT}/{name}_aligned.ply", cc)
        # heatmap (top view), clip color at 0.3 m
        P = np.asarray(cc.points)
        sc = axes[k].scatter(P[:, 0], P[:, 1], c=np.clip(d, 0, 0.3), s=1.2,
                             cmap="turbo")
        axes[k].scatter(L[::20, 0], L[::20, 1], s=0.1, c="lightgray", alpha=0.3, zorder=0)
        axes[k].set_title(f"{name}  fit={reg.fitness:.2f} rmse={reg.inlier_rmse:.3f}m\n"
                          f"C2C median={np.median(d)*100:.1f}cm mean={d.mean()*100:.1f}cm")
        axes[k].set_aspect("equal", "box")
        plt.colorbar(sc, ax=axes[k], label="dist to lidar (m, clip .3)")
    fig.tight_layout(); fig.savefig(f"{OUT}/c2c_heatmaps.png", dpi=110); plt.close(fig)

    # histograms
    plt.figure(figsize=(10, 6))
    for name, d in dists.items():
        plt.hist(np.clip(d, 0, 0.5), bins=100, alpha=0.5, label=f"{name} (med {np.median(d)*100:.1f}cm)")
    plt.xlabel("cloud-to-cloud distance to lidar (m)"); plt.ylabel("count"); plt.legend()
    plt.title("A (MediaSDK equirect) vs B (FIORD fisheye) — recon error vs lidar")
    plt.savefig(f"{OUT}/c2c_hist.png", dpi=110); plt.close()

    with open(f"{OUT}/c2c_summary.txt", "w") as f:
        for name, s in summ:
            line = (f"{name}: align fitness={s['fitness']:.3f} inlier_rmse={s['inlier_rmse']:.3f}m | "
                    f"C2C mean={s['mean']*100:.1f}cm median={s['median']*100:.1f}cm "
                    f"rms={s['rms']*100:.1f}cm p90={s['p90']*100:.1f}cm")
            print(line); f.write(line + "\n")
    print("\nWROTE:", OUT, "(c2c_heatmaps.png, c2c_hist.png, *_aligned.ply, c2c_summary.txt)")


if __name__ == "__main__":
    main()
