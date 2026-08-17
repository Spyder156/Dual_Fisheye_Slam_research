#!/usr/bin/env python
"""Rerun recording of a SLAM run: tracker frames + aligned trajectories + tip
axes + error plots.

All trajectories are expressed in the SLAM world frame: GT (+cloud) is
gravity-aligned then yaw/origin-matched to the SLAM start; the IMU dead-reckon
path likewise. Tip triads (UP=blue FWD=green LEFT=red) ride every trajectory.

Usage:
  viz_slam_rerun.py <dataset_dir> <traj.csv> <out.rrd> [--gt gt.csv]
      [--colmap-model dir] [--trackviz dir] [--stride 4]
"""
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

R_ItoC0 = np.load("/home/raghav/workspace/INSV_STITCHING/SLAM/configs/insta360_oners/R_ItoC0_gt.npy")
FPS = 30000 / 1001
G_W_COLMAP = np.array([-0.23, -0.97, 0.085])  # accel rest reading (UP) in Home COLMAP world


def set_t(t):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("t", t)
    else:
        rr.set_time("t", duration=t)


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    ax = ax / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def read_cam0_rotations(model: Path):
    out = {}
    with open(model / "images.bin", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", fh.read(4))
            q = np.array(struct.unpack("<4d", fh.read(32)))
            fh.read(24)
            struct.unpack("<I", fh.read(4))
            nm = b""
            while True:
                c = fh.read(1)
                if c == b"\x00":
                    break
                nm += c
            np_ = struct.unpack("<Q", fh.read(8))[0]
            fh.read(24 * np_)
            nm = nm.decode()
            if nm.startswith("cam0/"):
                w_, x_, y_, z_ = q
                out[int(nm.split("/")[1].split(".")[0])] = np.array([
                    [1-2*(y_*y_+z_*z_), 2*(x_*y_-w_*z_), 2*(x_*z_+w_*y_)],
                    [2*(x_*y_+w_*z_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-w_*x_)],
                    [2*(x_*z_-w_*y_), 2*(y_*z_+w_*x_), 1-2*(x_*x_+y_*y_)]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("traj", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--colmap-model", type=Path, default=None)
    ap.add_argument("--colmap-scale", type=float, default=1.2967)
    ap.add_argument("--trackviz", type=Path, default=None)
    args = ap.parse_args()

    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="trajectories", contents="/world/**"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/cam0", name="camera", contents="/cam0/**"),
                rrb.TimeSeriesView(origin="/plots/err/rot_err_deg", name="rotation error [deg]"),
                rrb.TimeSeriesView(origin="/plots/err/reproj_rms_px", name="reprojection rms [px]"),
                row_shares=[3, 1, 1],
            ),
        )
    )
    rr.init("insv_slam_v2", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)

    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti_ = imu["t"]
    gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1)
    g_world = np.array([0, 0, 9.81])
    up_body = acc[ti_ < 1.0].mean(0)
    up_body = up_body / np.linalg.norm(up_body)
    f_I = R_ItoC0.T @ np.array([0, 0, 1.0])

    d = np.genfromtxt(args.traj, delimiter=",", names=True)
    P = np.stack([d["px"], d["py"], d["pz"]], 1)
    T = d["t"]
    rots = Rotation.from_quat(np.stack([d["qx"], d["qy"], d["qz"], d["qw"]], 1))
    R_ItoG = [r.as_matrix() for r in rots]  # JPL-as-Hamilton = R_ItoG by construction
    f0 = R_ItoG[0] @ f_I
    yaw_sl0 = np.arctan2(f0[1], f0[0])

    def triad(name, org, R_wI):
        up = R_wI @ up_body
        fw = R_wI @ f_I
        fw = fw - up * (fw @ up)
        n = np.linalg.norm(fw)
        if n > 1e-6:
            fw /= n
        left = np.cross(up, fw)
        rr.log(f"world/tip_axes/{name}", rr.Arrows3D(
            origins=[org] * 3, vectors=[up * 0.5, fw * 0.5, left * 0.5],
            colors=[[60, 130, 255], [30, 200, 90], [230, 70, 70]]))

    # SLAM trajectory + tip axes
    for i in range(1, len(P)):
        set_t(T[i])
        rr.log("world/traj", rr.LineStrips3D([P[: i + 1]], colors=[[42, 120, 214]]))
        rr.log("world/pose", rr.Points3D([P[i]], colors=[[42, 120, 214]], radii=0.05))
        if i % 3 == 0:
            triad("slam", P[i], R_ItoG[i])

    # IMU dead-reckon: gravity-aligned, yaw+origin matched to SLAM start
    R = rot_a_to_b(up_body, np.array([0, 0, 1.0]))
    fi0 = R @ f_I
    R = Rz(yaw_sl0 - np.arctan2(fi0[1], fi0[0])) @ R
    ba = acc[ti_ < 1.0].mean(0) - R.T @ g_world
    p = P[0].copy()
    v = np.zeros(3)
    path = [p.copy()]
    times = [ti_[0]]
    Rs_imu = [R.copy()]
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
        if np.linalg.norm(p - P[0]) > 40.0:
            print(f"imu path left 40m radius at t={ti_[k]:.1f}s; truncated")
            break
        if k % 100 == 0:
            path.append(p.copy())
            times.append(ti_[k + 1])
            Rs_imu.append(R.copy())
    path = np.array(path)
    for i in range(1, len(path)):
        set_t(times[i])
        rr.log("world/imu_traj", rr.LineStrips3D([path[: i + 1]], colors=[[235, 104, 52]]))
        if i % 3 == 0:
            triad("imu", path[i], Rs_imu[i])

    # GT (+cloud): gravity-align COLMAP world, yaw+origin match to SLAM start
    if args.gt is not None:
        g = np.genfromtxt(args.gt, delimiter=",", names=True)
        tg = (g["frame"] - 1) / FPS - 0.012
        Pg_raw = np.stack([g["x"], g["y"], g["z"]], 1)
        Ra_gt = rot_a_to_b(G_W_COLMAP, np.array([0, 0, 1.0]))
        yaw_off = 0.0
        gtR = {}
        if args.colmap_model is not None:
            gtR = read_cam0_rotations(args.colmap_model)
            fr0 = sorted(gtR)[0]
            fg0 = Ra_gt @ gtR[fr0].T @ R_ItoC0 @ f_I
            yaw_off = yaw_sl0 - np.arctan2(fg0[1], fg0[0])
        A = Rz(yaw_off) @ Ra_gt
        Pg = (A @ (Pg_raw - Pg_raw[0]).T).T + P[0]
        gt_frames = sorted(gtR)
        for i in range(1, len(Pg)):
            set_t(tg[i])
            rr.log("world/gt_traj", rr.LineStrips3D([Pg[: i + 1]], colors=[[27, 175, 122]]))
            if gtR and i % 2 == 0 and i < len(gt_frames):
                R_wI = A @ gtR[gt_frames[i]].T @ R_ItoC0
                triad("gt", Pg[i], R_wI)
        if args.colmap_model is not None:
            import viz_colmap_rerun as vcr
            pts, cols = vcr.read_points3d(args.colmap_model / "points3D.bin")
            pts_m = (A @ (pts * args.colmap_scale - Pg_raw[0]).T).T + P[0]
            set_t(tg[0])
            rr.log("world/colmap_cloud", rr.Points3D(pts_m, colors=cols, radii=0.006))

    # rotation error (filter yaw vs gyro yaw) + reprojection error plots
    yawf = np.unwrap([np.arctan2((Rm @ f_I)[1], (Rm @ f_I)[0]) for Rm in R_ItoG])
    Rg_ = R_ItoG[0].copy()
    k0 = np.searchsorted(ti_, T[0])
    yg, tg2 = [np.arctan2((Rg_ @ f_I)[1], (Rg_ @ f_I)[0])], [T[0]]
    for k in range(k0, np.searchsorted(ti_, T[-1]) - 1):
        th = gyr[k] * (ti_[k + 1] - ti_[k])
        a = np.linalg.norm(th)
        if a > 1e-12:
            kx = th / a
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            Rg_ = Rg_ @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)
        if k % 100 == 0:
            yg.append(np.arctan2((Rg_ @ f_I)[1], (Rg_ @ f_I)[0]))
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

    # camera panel
    if args.trackviz is not None:
        import csv as _csv
        with open(args.trackviz / "viz.csv") as fh:
            rows = list(_csv.DictReader(fh))
        for row in rows:
            img = cv2.imread(str(args.trackviz / row["file"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w > 1600:
                img = cv2.resize(img, (1600, int(h * 1600 / w)))
            set_t(float(row["t"]))
            rr.log("cam0/image", rr.Image(img[:, :, ::-1]).compress(jpeg_quality=75))
        print(f"logged {len(rows)} tracker frames")
    else:
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
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
