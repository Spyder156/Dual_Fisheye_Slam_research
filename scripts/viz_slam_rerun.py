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
    args = ap.parse_args()

    rr.init("insv_slam", spawn=False)
    rr.save(str(args.out))

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
    p = np.zeros(3)
    v = np.zeros(3)
    path = [p.copy()]
    times = [ti_[0]]
    for k in range(len(ti_) - 1):
        dt = ti_[k + 1] - ti_[k]
        a_w = R @ acc[k] - g_world
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        th = gyr[k] * dt
        ang = np.linalg.norm(th)
        if ang > 1e-12:
            kx = th / ang
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            R = R @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
        if np.linalg.norm(p) > 25.0:
            print(f"imu path left 25m radius at t={ti_[k]:.1f}s; truncated there")
            break
        if k % 100 == 0:
            path.append(p.copy())
            times.append(ti_[k + 1])
    path = np.array(path)
    for i in range(1, len(path)):
        set_t(times[i])
        rr.log("world/imu_traj", rr.LineStrips3D([path[: i + 1]], colors=[[235, 104, 52]]))

    if args.gt is not None:
        g = np.genfromtxt(args.gt, delimiter=",", names=True)
        Pg = np.stack([g["x"], g["y"], g["z"]], 1)
        Pg = Pg - Pg[0] + P[0]  # origin-shifted to filter start; orientation unaligned
        rr.log("world/gt_traj", rr.LineStrips3D([Pg], colors=[[27, 175, 122]]), static=True)

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
            rr.log("cam0/tracks", rr.Points2D(p1.reshape(-1, 2)[ok], colors=[[27, 175, 122]], radii=1.5))
        ppts = cv2.goodFeaturesToTrack(img, 250, 0.01, 8, mask=mask)
        prev = img
    print(f"saved {args.out} — open with: rerun {args.out}")


if __name__ == "__main__":
    main()
