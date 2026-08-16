#!/usr/bin/env python
"""Coordinate-convention debug: one static triad per chain + signed first-turn
direction + fisheye frames with claimed axes projected on them.

Triads (X=red, Y=green, Z=blue arrows) at separate origins, one per chain:
  GT | gyro | optical | SLAM  — each shows [up, forward@t0, left=up x fwd]
  as that chain believes them, in its own gravity-aligned frame.
All four should be identical. A mirrored chain shows LEFT flipped and its
signed first-turn yaw delta with opposite sign (printed in legend).

Usage: viz_axes_debug.py <dataset_dir> <slam_traj.csv> <gt.csv> <out.rrd>
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

FPS = 30000 / 1001
S = "/tmp/claude-1000/-home-raghav-workspace-INSV-STITCHING/3d7e92da-d72b-4eb8-b5c2-7050abc3ca19/scratchpad"

xi = 2.01493
fx, fy, cx, cy = 2989.56, 2989.35, 1471.94, 1446.21
k1, k2, p1, p2 = 0.22369747, -0.22588399, -0.00104794, -0.00129457


def mei_project(X):
    Xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    z = Xs[:, 2] + xi
    x, y = Xs[:, 0] / z, Xs[:, 1] / z
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


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


def yaw(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("slam", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti = imu["t"]
    gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1)
    R_ItoC0 = np.load(f"{S}/R_ItoC0_gt.npy")

    # ---------- GT chain ----------
    g = np.genfromtxt(args.gt, delimiter=",", names=True)
    tg = (g["frame"] - 1) / FPS - 0.012
    Pg = np.stack([g["x"], g["y"], g["z"]], 1)
    g_w = np.array([-0.23, -0.97, 0.085])
    Ra_gt = rot_a_to_b(-g_w, np.array([0, 0, 1.0]))
    Pg = (Ra_gt @ Pg.T).T
    Vg = np.gradient(Pg, tg, axis=0)
    speed = np.linalg.norm(Vg, axis=1)
    fwd_gt = Vg[np.argmax(speed > 0.2)]
    fwd_gt = fwd_gt / np.linalg.norm(fwd_gt)

    # pick the first-turn window from GT yaw: first |delta| > 45deg over 6s
    yaws_gt = np.unwrap(np.radians([yaw(v) for v in Vg]))
    ta, tb = None, None
    for i in range(len(tg) - 1):
        j = np.searchsorted(tg, tg[i] + 6.0)
        if j >= len(tg):
            break
        if abs(np.degrees(yaws_gt[j] - yaws_gt[i])) > 45:
            ta, tb = tg[i], tg[j]
            dyaw_gt = np.degrees(yaws_gt[j] - yaws_gt[i])
            break
    print(f"first-turn window: t=[{ta:.1f},{tb:.1f}]s, GT dyaw={dyaw_gt:+.0f} deg")

    # ---------- gyro chain ----------
    up_body = acc[ti < 1.0].mean(0)
    Ra_gy = rot_a_to_b(up_body, np.array([0, 0, 1.0]))  # imu(t0) -> aligned
    f_I = R_ItoC0.T @ np.array([0, 0, 1.0])
    def gyro_R(t0, t1):
        m = (ti >= t0) & (ti < t1)
        R = np.eye(3)
        ts_, ws_ = ti[m], gyr[m]
        for k in range(len(ts_) - 1):
            th = ws_[k] * (ts_[k + 1] - ts_[k])
            a_ = np.linalg.norm(th)
            if a_ < 1e-12:
                continue
            kx = th / a_
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            R = R @ (np.eye(3) + np.sin(a_) * K + (1 - np.cos(a_)) * K @ K)
        return R
    up_gy = Ra_gy @ up_body
    fwd_gy0 = Ra_gy @ f_I
    Rta = gyro_R(0, ta)
    Rtb = gyro_R(0, tb)
    dyaw_gy = yaw(Ra_gy @ (Rtb @ f_I)) - yaw(Ra_gy @ (Rta @ f_I))
    dyaw_gy = (dyaw_gy + 180) % 360 - 180

    # ---------- optical chain ----------
    up_cam = np.array([0, -1.0, 0])
    Ra_vo = rot_a_to_b(up_cam, np.array([0, 0, 1.0]))
    def vo_R(f0, f1):
        Rw = np.eye(3)
        for i in range(f0, f1 - 10, 5):
            a = cv2.imread(str(args.dataset / "cam0" / f"{i:06d}.jpg"), 0)
            b = cv2.imread(str(args.dataset / "cam0" / f"{i+10:06d}.jpg"), 0)
            if a is None or b is None:
                continue
            a2 = cv2.resize(a, None, fx=0.5, fy=0.5)
            b2 = cv2.resize(b, None, fx=0.5, fy=0.5)
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
            Rw = kabsch(b0[idx], b1[idx]) @ Rw
        return Rw  # maps cam(f0) vectors to cam(f1) frame
    fa, fb = int(ta * FPS) + 1, int(tb * FPS) + 1
    Rvo = vo_R(fa, fb)
    z0 = np.array([0, 0, 1.0])
    dyaw_vo = yaw(Ra_vo @ (Rvo.T @ z0)) - yaw(Ra_vo @ z0)
    dyaw_vo = (dyaw_vo + 180) % 360 - 180
    fwd_vo0 = Ra_vo @ z0

    # ---------- SLAM chain ----------
    d = np.genfromtxt(args.slam, delimiter=",", names=True)
    ts = d["t"]
    Q = np.stack([d["qx"], d["qy"], d["qz"], d["qw"]], 1)
    rots = Rotation.from_quat(Q)
    g_world = np.array([0, 0, 9.81])
    am = acc[np.searchsorted(ti, ts[len(ts) // 2])]
    Rm = rots[len(ts) // 2].as_matrix()
    flip = np.linalg.norm(Rm @ am - g_world) < np.linalg.norm(Rm.T @ am - g_world)
    def slam_fwd(t):
        k = np.searchsorted(ts, t)
        k = min(max(k, 0), len(ts) - 1)
        R = rots[k].as_matrix() if flip else rots[k].as_matrix().T
        return R @ f_I
    dyaw_sl = yaw(slam_fwd(tb)) - yaw(slam_fwd(ta))
    dyaw_sl = (dyaw_sl + 180) % 360 - 180
    fwd_sl0 = slam_fwd(ts[0])
    up_sl = np.array([0, 0, 1.0])

    # ---------- log triads ----------
    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/triads", name="per-chain axes (should all match)", contents="/triads/**"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/frame_axes", name="claimed axes on fisheye", contents="/frame_axes/**"),
                rrb.TextDocumentView(origin="/legend", name="numbers"),
            ),
            column_shares=[2, 1],
        )
    )
    rr.init("axes_debug", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)

    chains = [
        ("gt", np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), fwd_gt, dyaw_gt),
        ("gyro", np.array([3, 0, 0.0]), up_gy / np.linalg.norm(up_gy), fwd_gy0 / np.linalg.norm(fwd_gy0), dyaw_gy),
        ("optical", np.array([6, 0, 0.0]), Ra_vo @ up_cam, fwd_vo0 / np.linalg.norm(fwd_vo0), dyaw_vo),
        ("slam", np.array([9, 0, 0.0]), up_sl, fwd_sl0 / np.linalg.norm(fwd_sl0), dyaw_sl),
    ]
    legend = [f"first-turn window t=[{ta:.1f},{tb:.1f}]s — signed yaw delta per chain:"]
    for name, org, up, fwd, dy in chains:
        fwd_h = fwd - up * (fwd @ up)
        fwd_h = fwd_h / (np.linalg.norm(fwd_h) + 1e-9)
        left = np.cross(up, fwd_h)
        rr.log(f"triads/{name}", rr.Arrows3D(
            origins=[org, org, org],
            vectors=[up * 1.2, fwd_h * 1.2, left * 1.2],
            colors=[[60, 130, 255], [30, 200, 90], [230, 70, 70]],
            labels=[f"{name}:UP", f"{name}:FWD@t0", f"{name}:LEFT"]), static=True)
        legend.append(f"  {name:8s} dyaw = {dy:+7.1f} deg   (same sign as gt = same convention)")
    rr.log("legend", rr.TextDocument("\n".join(legend)), static=True)
    print("\n".join(legend))

    # ---------- fisheye frames with projected axes ----------
    for tag, fr in [("start", 200), ("mid_turn", int((ta + tb) / 2 * FPS))]:
        img = cv2.imread(str(args.dataset / "cam0" / f"{fr:06d}.jpg"))
        if img is None:
            continue
        axes_I = np.eye(3)  # imu axes
        colors = [(60, 60, 235), (60, 200, 60), (235, 120, 40)]
        names = ["imu_X(up?)", "imu_Y", "imu_Z"]
        ctr3 = np.array([[0, 0, 1.0]])
        c2 = mei_project(ctr3)[0]
        for k in range(3):
            v_cam = R_ItoC0 @ axes_I[k]
            tip3 = np.array([0, 0, 1.0]) + 0.35 * v_cam
            p2 = mei_project(tip3[None, :])[0]
            cv2.arrowedLine(img, tuple(c2.astype(int)), tuple(p2.astype(int)), colors[k], 12, tipLength=0.2)
            cv2.putText(img, names[k], tuple((p2 + 30).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, colors[k], 5)
        cv2.putText(img, f"{tag} f{fr}: IMU axes projected via R_ItoC0 (blue=imu_X, should point at CEILING side)",
                    (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        img = cv2.resize(img, None, fx=0.35, fy=0.35)
        rr.log(f"frame_axes/{tag}", rr.Image(img[:, :, ::-1]).compress(jpeg_quality=80), static=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
