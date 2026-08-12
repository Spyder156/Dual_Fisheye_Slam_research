#!/usr/bin/env python3
"""
Robust COLMAP->lidar registration (v2). The v1 failure was scale: a sparse SfM
cloud and a dense lidar have different point *distributions*, so a single
robust-radius scale guess was wrong. Here we SWEEP scale, run FPFH+RANSAC
global registration at each, keep the best by inlier fitness, then refine with
multiscale point-to-plane ICP. Reports inlier_rmse @5cm (the real trust signal).

Anchor trick: A and B are already sim3-aligned (pose_compare). So we align the
*trusted* dense-ish reference (B) to lidar ONCE, and also align A independently
as a cross-check.

Outputs -> outputs/viz/c2c_v2/
"""
import os, copy, numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FI = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom"
OUT = f"{ROOT}/outputs/viz/c2c_v2"; os.makedirs(OUT, exist_ok=True)


def load_lidar(voxel=0.03):
    import pye57
    e = pye57.E57(f"{FI}/meetingroom.e57")
    d = e.read_scan(0, ignore_missing_fields=True, colors=False, intensity=False)
    P = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1)
    P = P[np.isfinite(P).all(1)]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    return pc.voxel_down_sample(voxel)


def load_colmap(path):
    xyz = []
    with open(f"{path}/points3D.txt") as f:
        for l in f:
            if l.startswith("#") or not l.strip():
                continue
            p = l.split(); xyz.append(list(map(float, p[1:4])))
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.array(xyz))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pc


def prep(pc, voxel):
    d = pc.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
    return d, f


def ransac(src_d, src_f, dst_d, dst_f, voxel):
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, dst_d, src_f, dst_f, True, voxel*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel*1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))


def robust_radius(pc):
    P = np.asarray(pc.points); c = np.median(P, 0)
    return np.median(np.linalg.norm(P - c, axis=1))


def register(colmap, lidar, lidar_d, lidar_f, voxel=0.05, tag=""):
    s0 = robust_radius(lidar) / robust_radius(colmap)
    best = None
    for k in np.linspace(0.6, 1.6, 11):
        s = s0 * k
        c = copy.deepcopy(colmap)
        c.scale(s, center=c.get_center())
        cd, cf = prep(c, voxel)
        r = ransac(cd, cf, lidar_d, lidar_f, voxel)
        if best is None or r.fitness > best[3].fitness:
            best = (s, c, cd, r)
        print(f"  [{tag}] scale x{k:.2f} (s={s:.4f}) fitness={r.fitness:.3f} rmse={r.inlier_rmse:.3f}")
    s, c, cd, r = best
    # multiscale point-to-plane ICP refine
    T = r.transformation
    for vv, th in [(0.05, 0.15), (0.03, 0.08), (0.02, 0.05)]:
        cds = c.voxel_down_sample(vv)
        cds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vv*2, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(
            cds, lidar, th, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
        T = reg.transformation
    c.transform(T)
    return s, c, reg


def c2c_stats(colmap_aligned, lidar):
    A = np.asarray(colmap_aligned.points); L = np.asarray(lidar.points)
    d, _ = cKDTree(L).query(A, k=1)
    return d


def main():
    lidar = load_lidar(0.03)
    lidar.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    lidar_d, lidar_f = prep(lidar, 0.05)
    L = np.asarray(lidar.points)

    arms = {"B_FIORD": f"{ROOT}/outputs/viz/clouds/B_txt",
            "A_MediaSDK": f"{ROOT}/outputs/benchmark/armA_mediasdk_equirect/sparse/0"}
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for i, (name, path) in enumerate(arms.items()):
        print(f"== {name} ==")
        cm = load_colmap(path)
        s, aligned, reg = register(cm, lidar, lidar_d, lidar_f, tag=name)
        d = c2c_stats(aligned, lidar)
        results[name] = dict(scale=s, fitness=reg.fitness, rmse=reg.inlier_rmse,
                             med=np.median(d), mean=d.mean(), p90=np.percentile(d, 90), d=d)
        o3d.io.write_point_cloud(f"{OUT}/{name}_aligned.ply", aligned)
        P = np.asarray(aligned.points)
        sc = axes[i].scatter(P[:, 0], P[:, 1], c=np.clip(d, 0, 0.2), s=1.2, cmap="turbo")
        axes[i].scatter(L[::20, 0], L[::20, 1], s=0.1, c="lightgray", alpha=0.3, zorder=0)
        axes[i].set_title(f"{name}  ICP rmse={reg.inlier_rmse*100:.1f}cm fit={reg.fitness:.2f}\n"
                          f"C2C med={np.median(d)*100:.1f}cm p90={np.percentile(d,90)*100:.1f}cm")
        axes[i].set_aspect("equal", "box"); plt.colorbar(sc, ax=axes[i], label="dist (m)")
    fig.tight_layout(); fig.savefig(f"{OUT}/c2c_heatmaps.png", dpi=110); plt.close(fig)

    plt.figure(figsize=(10, 6))
    for name, r in results.items():
        plt.hist(np.clip(r["d"], 0, 0.3), bins=100, alpha=0.5,
                 label=f"{name} med {r['med']*100:.1f}cm")
    plt.legend(); plt.xlabel("C2C dist to lidar (m)"); plt.ylabel("count")
    plt.title("Recon error vs lidar (v2 registration)")
    plt.savefig(f"{OUT}/c2c_hist.png", dpi=110); plt.close()

    with open(f"{OUT}/summary.txt", "w") as f:
        for name, r in results.items():
            line = (f"{name}: ICP_inlier_rmse={r['rmse']*100:.1f}cm fitness={r['fitness']:.3f} scale={r['scale']:.4f} | "
                    f"C2C median={r['med']*100:.1f}cm mean={r['mean']*100:.1f}cm p90={r['p90']*100:.1f}cm")
            print(line); f.write(line + "\n")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
