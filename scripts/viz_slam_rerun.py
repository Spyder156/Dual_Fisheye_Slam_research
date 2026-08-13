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

    # imu magnitudes
    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    for i in range(0, len(imu["t"]), 50):
        set_t(imu["t"][i])
        rr.log("imu/gyro_mag", rr.Scalars(float(np.hypot(np.hypot(imu["gx"][i], imu["gy"][i]), imu["gz"][i]))))
        rr.log("imu/accel_mag", rr.Scalars(float(np.hypot(np.hypot(imu["ax"][i], imu["ay"][i]), imu["az"][i]))))

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
