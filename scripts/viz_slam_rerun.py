#!/usr/bin/env python
"""Rerun recording of a SLAM run: fisheye frames + tracked points + trajectory + IMU.

Logs to a .rrd file — open with `rerun <file.rrd>`.
- world/traj      : estimated trajectory (line + current pose)
- cam0            : fisheye frames (downsampled) with KLT-tracked points overlaid
- imu/gyro|accel  : signal magnitudes
Run after every SLAM run:
  python scripts/viz_slam_rerun.py <dataset_dir> <traj.csv> <out.rrd> [--stride 4]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb


def set_t(t):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("t", t)
    else:
        rr.set_time("t", duration=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("traj", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stride", type=int, default=4, help="log every Nth frame")
    ap.add_argument("--scale", type=float, default=0.25, help="image downscale for logging")
    ap.add_argument("--gt", type=Path, default=None, help="GT csv (frame,x,y,z) to overlay, own frame")
    ap.add_argument("--colmap-model", type=Path, default=None, help="COLMAP sparse model dir for cloud overlay")
    ap.add_argument("--colmap-scale", type=float, default=1.2967)
    args = ap.parse_args()

    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="trajectories", contents="/world/**"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/cam0", name="camera", contents="/cam0/**"),
                rrb.TimeSeriesView(origin="/plots/err", name="errors", contents="/plots/err/**"),
                rrb.TimeSeriesView(origin="/plots/imu", name="imu", contents="/plots/imu/**"),
                row_shares=[3, 1, 1],
            ),
        )
    )
    rr.init("insv_slam_v2", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)

    # trajectory
    d = np.genfromtxt(args.traj, delimiter=",", names=True)
    P = np.stack([d["px"], d["py"], d["pz"]], 1)
    T = d["t"]
    for i in range(1, len(P)):
        set_t(T[i])
        rr.log("world/traj", rr.LineStrips3D([P[: i + 1]], colors=[[42, 120, 214]]))
        rr.log("world/pose", rr.Points3D([P[i]], colors=[[235, 104, 52]], radii=0.05))

    # IMU-only trajectory: starts at t=0 (video start), seeded stationary and
    # gravity-aligned (yaw arbitrary), pure dead-reckoning — fully independent
    # of the filter. Drawing stops if it drifts outside a 25m radius so the
    # 3D view stays usable (dead-reckoning blows up; the early path is the
    # informative part).
    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti_ = imu["t"]
    gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1)
    g_world = np.array([0, 0, 9.81])
    g0 = acc[ti_ < 0.5].mean(0)
    g0 = g0 / np.linalg.norm(g0)
    # R_ItoG seed: rotate measured gravity dir onto +z (minimal rotation)
    zw = np.array([0, 0, 1.0])
    ax = np.cross(g0, zw)
    s = np.linalg.norm(ax)
    c = float(g0 @ zw)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        ax = ax / s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        ang0 = np.arctan2(s, c)
        R = np.eye(3) + np.sin(ang0) * K + (1 - np.cos(ang0)) * K @ K
    ba = acc[ti_ < 1.0].mean(0) - R.T @ g_world  # standstill accel bias estimate
    p = np.zeros(3)
    v = np.zeros(3)
    path = [p.copy()]
    times = [ti_[0]]
    for k in range(len(ti_) - 1):
        dt = ti_[k + 1] - ti_[k]
        a_w = R @ (acc[k] - ba) - g_world
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        th = gyr[k] * dt
        ang = np.linalg.norm(th)
        if ang > 1e-12:
            kx = th / ang
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            R = R @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
        if np.linalg.norm(p) > 40.0:
            print(f"imu path left 40m radius at t={ti_[k]:.1f}s; truncated there")
            break
        if k % 100 == 0:
            path.append(p.copy())
            times.append(ti_[k + 1])
    path = np.array(path)
    for i in range(1, len(path)):
        set_t(times[i])
        rr.log("world/imu_traj", rr.LineStrips3D([path[: i + 1]], colors=[[235, 104, 52]]))

    # scalar plots: imu magnitudes, rotation error (filter vs gyro yaw), reprojection error
    wmag = np.linalg.norm(gyr, axis=1)
    amag = np.linalg.norm(acc, axis=1)
    for k in range(0, len(ti_), 100):
        set_t(ti_[k])
        rr.log("plots/imu/gyro_rads", rr.Scalars(float(wmag[k])))
        rr.log("plots/imu/accel_ms2", rr.Scalars(float(amag[k])))
    R_ItoG = [r.as_matrix() if flip else r.as_matrix().T for r in rots]
    fwd = np.array([1.0, 0, 0])
    yawf = np.unwrap([np.arctan2((Rm_ @ fwd)[1], (Rm_ @ fwd)[0]) for Rm_ in R_ItoG])
    Rg_ = R_ItoG[0].copy()
    k0 = np.searchsorted(ti_, T[0])
    yg, tg2 = [np.arctan2((Rg_ @ fwd)[1], (Rg_ @ fwd)[0])], [T[0]]
    for k in range(k0, np.searchsorted(ti_, T[-1]) - 1):
        dtk = ti_[k + 1] - ti_[k]
        th = gyr[k] * dtk
        a = np.linalg.norm(th)
        if a > 1e-12:
            kx = th / a
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            Rg_ = Rg_ @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)
        if k % 100 == 0:
            yg.append(np.arctan2((Rg_ @ fwd)[1], (Rg_ @ fwd)[0]))
            tg2.append(ti_[k])
    yg = np.unwrap(yg)
    yi = np.interp(tg2, T, yawf)
    err = np.degrees(np.abs((yi - yi[0]) - (yg - yg[0])))
    for tt_, e in zip(tg2, err):
        set_t(tt_)
        rr.log("plots/err/rot_err_deg", rr.Scalars(float(e)))
    stats_path = Path(str(args.traj) + ".stats.csv")
    if stats_path.exists():
        st = np.genfromtxt(stats_path, delimiter=",", names=True)
        for k in range(len(st["t"])):
            if st["reproj_rms_px"][k] >= 0:
                set_t(st["t"][k])
                rr.log("plots/err/reproj_rms_px", rr.Scalars(float(st["reproj_rms_px"][k])))

    if args.gt is not None:
        g = np.genfromtxt(args.gt, delimiter=",", names=True)
        Pg = np.stack([g["x"], g["y"], g["z"]], 1)
        Pg = Pg - Pg[0] + P[0]  # origin-shifted to filter start; orientation unaligned
        rr.log("world/gt_traj", rr.LineStrips3D([Pg], colors=[[27, 175, 122]]), static=True)
        if args.colmap_model is not None:
            import viz_colmap_rerun as vcr
            pts, cols = vcr.read_points3d(args.colmap_model / "points3D.bin")
            g0 = np.stack([g["x"], g["y"], g["z"]], 1)[0]
            pts_m = pts * args.colmap_scale - g0 + P[0]
            rr.log("world/colmap_cloud", rr.Points3D(pts_m, colors=cols, radii=0.006), static=True)

    # frames + KLT tracks (python-side, for visualization of what is trackable)
    frames = np.genfromtxt(args.dataset / "frames.csv", delimiter=",", names=True)
    fids = frames["frame"].astype(int)
    fts = frames["t"]
    prev, ppts = None, None
    for k in range(0, len(fids), args.stride):
        img = cv2.imread(str(args.dataset / "cam0" / f"{fids[k]:06d}.jpg"), 0)
        if img is None:
            continue
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
        h, w = img.shape
        set_t(fts[k])
        rr.log("cam0/image", rr.Image(img).compress(jpeg_quality=70))
        mask = np.zeros_like(img)
        cv2.circle(mask, (w // 2, h // 2), int(0.46 * w), 255, -1)
        if prev is not None and ppts is not None and len(ppts):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, ppts, None, winSize=(15, 15), maxLevel=4)
            ok = st.ravel().astype(bool)
            rr.log("cam0/image/tracks", rr.Points2D(p1.reshape(-1, 2)[ok], colors=[[27, 175, 122]], radii=1.5))
        ppts = cv2.goodFeaturesToTrack(img, 250, 0.01, 8, mask=mask)
        prev = img
    print(f"saved {args.out} — open with: rerun {args.out}")


if __name__ == "__main__":
    main()
