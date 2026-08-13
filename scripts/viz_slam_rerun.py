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

    # IMU-only trajectory: seeded once from the first filter pose, then pure
    # dead-reckoning for the whole run — its own independent path, own color.
    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti_ = imu["t"]
    gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1)
    from scipy.spatial.transform import Rotation
    Q = np.stack([d["qx"], d["qy"], d["qz"], d["qw"]], 1)
    rots = Rotation.from_quat(Q)
    g_world = np.array([0, 0, 9.81])
    am = acc[np.searchsorted(ti_, T[len(T) // 2])]
    R_mid = rots[len(T) // 2].as_matrix()
    flip = np.linalg.norm(R_mid @ am - g_world) < np.linalg.norm(R_mid.T @ am - g_world)
    R = rots[0].as_matrix() if flip else rots[0].as_matrix().T  # R_ItoG at t0
    p = P[0].copy()
    v = np.gradient(P, T, axis=0)[0].copy()
    m = (ti_ >= T[0]) & (ti_ <= T[-1])
    ts_, ws_, as_ = ti_[m], gyr[m], acc[m]
    path = [p.copy()]
    times = [ts_[0]]
    for k in range(len(ts_) - 1):
        dt = ts_[k + 1] - ts_[k]
        a_w = R @ as_[k] - g_world
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        th = ws_[k] * dt
        ang = np.linalg.norm(th)
        if ang > 1e-12:
            kx = th / ang
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            R = R @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
        if k % 100 == 0:
            path.append(p.copy())
            times.append(ts_[k + 1])
    path = np.array(path)
    for i in range(1, len(path)):
        set_t(times[i])
        rr.log("world/imu_traj", rr.LineStrips3D([path[: i + 1]], colors=[[235, 104, 52]]))

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
