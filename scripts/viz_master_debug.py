#!/usr/bin/env python
"""Master debug viz: 4 trajectories evolving together + live axes triad at each
tip + camera feed, all yaw/origin-aligned at start on one timeline.

Chains: gt (green path) | gyro+GTspeed (yellow) | optical VO (purple) | SLAM (blue).
Triads at tips: UP=blue, FWD=green, LEFT=red arrows.

Usage: viz_master_debug.py <dataset_dir> <slam_traj.csv> <gt.csv> <model_dir> <out.rrd>
"""
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

FPS = 30000 / 1001
R_ItoC0 = np.load("/home/raghav/workspace/INSV_STITCHING/SLAM/configs/insta360_oners/R_ItoC0_gt.npy")

xi = 2.01493
fx, fy, cx, cy = 2989.56, 2989.35, 1471.94, 1446.21
k1, k2, p1, p2 = 0.22369747, -0.22588399, -0.00104794, -0.00129457


def unproject(uv):
    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(12):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    r2 = x * x + y * y
    fac = (xi + np.sqrt(np.maximum(1 + (1 - xi * xi) * r2, 0))) / (1 + r2)
    b = np.stack([fac * x, fac * y, fac - xi], 1)
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def kabsch(A, B):
    U, S_, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


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


def yaw_align_R(R_wI, f_I):
    f = R_wI @ f_I
    return Rz(-np.arctan2(f[1], f[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("slam", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti = imu["t"]
    gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1)
    up_body = acc[ti < 1.0].mean(0)
    up_body /= np.linalg.norm(up_body)
    f_I = R_ItoC0.T @ np.array([0, 0, 1.0])

    # ---- GT: positions + rotations ----
    g = np.genfromtxt(args.gt, delimiter=",", names=True)
    tg = (g["frame"] - 1) / FPS - 0.012
    Pg_raw = np.stack([g["x"], g["y"], g["z"]], 1)
    g_w = np.array([-0.23, -0.97, 0.085])
    Ra_gt = rot_a_to_b(g_w, np.array([0, 0, 1.0]))
    cam0R = {}
    with open(args.model / "images.bin", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", fh.read(4))
            q_ = np.array(struct.unpack("<4d", fh.read(32)))
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
                w_, x_, y_, z_ = q_
                cam0R[int(nm.split("/")[1].split(".")[0])] = np.array([
                    [1-2*(y_*y_+z_*z_), 2*(x_*y_-w_*z_), 2*(x_*z_+w_*y_)],
                    [2*(x_*y_+w_*z_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-w_*x_)],
                    [2*(x_*z_-w_*y_), 2*(y_*z_+w_*x_), 1-2*(x_*x_+y_*y_)]])
    gt_frames = np.array(sorted(cam0R))
    gt_times = (gt_frames - 1) / FPS - 0.012
    def gt_R_wI(k):
        return Ra_gt @ cam0R[gt_frames[k]].T @ R_ItoC0
    A_gt = yaw_align_R(gt_R_wI(0), f_I)
    Pg = (A_gt @ (Ra_gt @ (Pg_raw - Pg_raw[0]).T)).T
    Vg = np.gradient((Ra_gt @ Pg_raw.T).T, tg, axis=0)
    speed = np.linalg.norm(Vg, axis=1)

    # ---- gyro chain ----
    Ra_gy = rot_a_to_b(up_body, np.array([0, 0, 1.0]))
    gy = {}
    Rg = np.eye(3)
    grid = np.arange(0, ti[-1], 0.1)
    gi = 0
    for k in range(len(ti) - 1):
        if gi < len(grid) and ti[k] >= grid[gi]:
            gy[round(grid[gi], 1)] = Ra_gy @ Rg
            gi += 1
        th = gyr[k] * (ti[k + 1] - ti[k])
        a_ = np.linalg.norm(th)
        if a_ > 1e-12:
            kx = th / a_
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            Rg = Rg @ (np.eye(3) + np.sin(a_) * K + (1 - np.cos(a_)) * K @ K)
    A_gy = yaw_align_R(gy[min(gy)], f_I)

    # ---- optical chain: rotations + translation dirs ----
    Ra_vo = Ra_gy @ R_ItoC0.T
    vo_R, vo_tdir = {}, {}
    Rw = np.eye(3)
    mask = None
    for i in range(1, 2540 - 10, 10):
        a = cv2.imread(str(args.dataset / "cam0" / f"{i:06d}.jpg"), 0)
        b = cv2.imread(str(args.dataset / "cam0" / f"{i+10:06d}.jpg"), 0)
        if a is None or b is None:
            continue
        a2 = cv2.resize(a, None, fx=0.5, fy=0.5)
        b2 = cv2.resize(b, None, fx=0.5, fy=0.5)
        if mask is None:
            mask = np.zeros_like(a2)
            cv2.circle(mask, (int(cx / 2), int(cy / 2)), 660, 255, -1)
        pts = cv2.goodFeaturesToTrack(a2, 200, 0.01, 12, mask=mask)
        if pts is None or len(pts) < 40:
            continue
        q2, st, _ = cv2.calcOpticalFlowPyrLK(a2, b2, pts, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel().astype(bool)
        if ok.sum() < 40:
            continue
        b0 = unproject(pts.reshape(-1, 2)[ok] * 2)
        b1 = unproject(q2.reshape(-1, 2)[ok] * 2)
        idx = np.arange(len(b0))
        for _ in range(3):
            R = kabsch(b0[idx], b1[idx])
            err = np.linalg.norm(b1 - b0 @ R.T, axis=1)
            idx = np.where(err < np.percentile(err, 70))[0]
        R = kabsch(b0[idx], b1[idx])
        Arows = np.cross(b1[idx], (R @ b0[idx].T).T)
        _, _, Vt = np.linalg.svd(Arows)
        td = Vt[-1]
        flow = b1[idx] - (R @ b0[idx].T).T
        s1 = np.median(np.einsum("ij,ij->i", flow,
                                 np.cross(np.cross(b1[idx], td[None, :].repeat(len(idx), 0)), b1[idx])))
        if s1 < 0:
            td = -td
        t_ = (i - 1 + 10) / FPS
        vo_tdir[t_] = (Rw.T @ td).copy()   # in cam0(t0) frame
        Rw = R @ Rw
        vo_R[t_] = Ra_vo @ Rw.T @ R_ItoC0  # body -> aligned
    A_vo = yaw_align_R(vo_R[min(vo_R)], f_I)

    # ---- SLAM ----
    d = np.genfromtxt(args.slam, delimiter=",", names=True)
    ts = d["t"]
    Ps_raw = np.stack([d["px"], d["py"], d["pz"]], 1)
    rots = Rotation.from_quat(np.stack([d["qx"], d["qy"], d["qz"], d["qw"]], 1))
    # JPL [x,y,z,w] of R_GtoI read as Hamilton = its transpose = R_ItoG. Always.
    def slam_R_wI(k):
        return rots[k].as_matrix()
    A_sl = yaw_align_R(slam_R_wI(0), f_I)
    Ps = (A_sl @ (Ps_raw - Ps_raw[0]).T).T

    # ---- log ----
    bp = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="trajectories + live axes", contents="/world/**"),
        rrb.Spatial2DView(origin="/cam0", name="camera", contents="/cam0/**"),
        column_shares=[3, 1]))
    rr.init("master_debug", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)
    def set_t(t):
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("t", t)
        else:
            rr.set_time("t", duration=t)

    def triad(name, org, R_wI, A):
        up = A @ (R_wI @ up_body)
        fwd = A @ (R_wI @ f_I)
        fwd_h = fwd - up * (fwd @ up)
        n = np.linalg.norm(fwd_h)
        if n > 1e-6:
            fwd_h /= n
        left = np.cross(up, fwd_h)
        rr.log(f"world/tip_axes/{name}", rr.Arrows3D(
            origins=[org] * 3, vectors=[up * 0.5, fwd_h * 0.5, left * 0.5],
            colors=[[60, 130, 255], [30, 200, 90], [230, 70, 70]]))

    colors = {"gt": [27, 175, 122], "gyro": [235, 161, 0], "optical": [150, 80, 220], "slam": [42, 120, 214]}
    # gyro + optical paths built incrementally with GT speed
    p_gy = np.zeros(3)
    path_gy = [p_gy.copy()]
    p_vo = np.zeros(3)
    path_vo = [p_vo.copy()]
    vo_keys = sorted(vo_R)
    vo_i = 0
    path_gt, path_sl = [], []
    last_gy_key = min(gy)
    for t in np.arange(0, 85, 0.2):
        set_t(t)
        # GT
        k = int(np.clip(np.searchsorted(gt_times, t), 0, len(gt_frames) - 1))
        if gt_times[0] <= t:
            path_gt.append(Pg[k])
            rr.log("world/traj/gt", rr.LineStrips3D([np.array(path_gt)], colors=[colors["gt"]]))
            triad("gt", Pg[k], gt_R_wI(k), A_gt)
        # gyro (advance position by heading x GT speed)
        gk = round(np.floor(t * 10) / 10, 1)
        if gk in gy:
            R = gy[gk]
            fwd = A_gy @ (R @ f_I)
            v = np.interp(t, tg, speed)
            p_gy = p_gy + fwd * v * 0.2
            path_gy.append(p_gy.copy())
            rr.log("world/traj/gyro", rr.LineStrips3D([np.array(path_gy)], colors=[colors["gyro"]]))
            triad("gyro", p_gy, R, A_gy)
        # optical
        while vo_i < len(vo_keys) and vo_keys[vo_i] <= t:
            tk = vo_keys[vo_i]
            v = np.interp(tk, tg, speed)
            step_dir = A_vo @ (Ra_vo @ vo_tdir[tk])
            p_vo = p_vo + step_dir * v * (10 / FPS)
            path_vo.append(p_vo.copy())
            vo_i += 1
        if len(path_vo) > 1:
            rr.log("world/traj/optical", rr.LineStrips3D([np.array(path_vo)], colors=[colors["optical"]]))
            lastk = vo_keys[max(vo_i - 1, 0)]
            triad("optical", p_vo, vo_R[lastk], A_vo)
        # SLAM
        if ts[0] <= t <= ts[-1]:
            k = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
            path_sl.append(Ps[k])
            rr.log("world/traj/slam", rr.LineStrips3D([np.array(path_sl)], colors=[colors["slam"]]))
            triad("slam", Ps[k], slam_R_wI(k), A_sl)
    for i in range(1, 2540, 8):
        set_t((i - 1) / FPS - 0.012)
        img = cv2.imread(str(args.dataset / "cam0" / f"{i:06d}.jpg"), 0)
        if img is not None:
            rr.log("cam0/image", rr.Image(cv2.resize(img, None, fx=0.15, fy=0.15)).compress(jpeg_quality=70))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
