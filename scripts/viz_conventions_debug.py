#!/usr/bin/env python
"""Convention-debug Rerun: independent single-modality trajectories side by side.

Trajectories (all yaw+origin aligned to GT at start, own shape otherwise):
  green  = COLMAP GT
  blue   = SLAM (OpenVINS)
  purple = pure optical VO (chained Kabsch rotation + epipolar translation dir,
           flat per-step scale = GT average speed; mono has no scale)
  yellow = gyro-orientation path (gyro headings x GT speed profile)
  orange = IMU dead-reckon (accel+gyro only)
Plots: yaw-vs-time of each chain, mapped through the SAME claimed conventions.
A sign/mirror convention bug shows as one curve running inverted or reflected.

Usage: viz_conventions_debug.py <dataset_dir> <slam_traj.csv> <gt.csv> <out.rrd>
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


def set_t(t):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("t", t)
    else:
        rr.set_time("t", duration=t)


def yaw_of(v):
    return np.arctan2(v[1], v[0])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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


def align_start(P, gt0, gtdir):
    """origin to gt0, initial horizontal direction to gtdir (yaw only)."""
    d = None
    for i in range(1, len(P)):
        step = P[i] - P[0]
        if np.linalg.norm(step[:2]) > 0.2:
            d = step
            break
    if d is None:
        return P - P[0] + gt0
    a = yaw_of(gtdir) - yaw_of(d)
    return (Rz(a) @ (P - P[0]).T).T + gt0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("slam", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    # GT
    g = np.genfromtxt(args.gt, delimiter=",", names=True)
    tg = (g["frame"] - 1) / FPS - 0.012
    Pg = np.stack([g["x"], g["y"], g["z"]], 1)
    # gravity-align the COLMAP world (its axes are arbitrary): measured gravity dir -> -z
    g_w = np.array([-0.23, -0.97, 0.085])
    Rg_align = rot_a_to_b(-g_w, np.array([0, 0, 1.0]))
    Pg = (Rg_align @ Pg.T).T
    Vg = np.gradient(Pg, tg, axis=0)
    speed = np.linalg.norm(Vg, axis=1)
    gt0, gtdir = Pg[0], Vg[np.argmax(speed > 0.2)]

    # SLAM
    d = np.genfromtxt(args.slam, delimiter=",", names=True)
    ts = d["t"]
    Ps = np.stack([d["px"], d["py"], d["pz"]], 1)
    Q = np.stack([d["qx"], d["qy"], d["qz"], d["qw"]], 1)

    # IMU
    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti = imu["t"]
    gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1)

    R_ItoC0 = np.load(f"{S}/R_ItoC0_gt.npy")
    f_I = R_ItoC0.T @ np.array([0, 0, 1.0])

    # ---- pure optical VO (cam0), gap 10, step 5 ----
    GAP, STEP = 10, 5
    Rw = np.eye(3)
    p = np.zeros(3)
    vo_pts, vo_t, vo_yaw = [p.copy()], [0.0], []
    mask = None
    for i in range(1, 2540 - GAP, STEP):
        a = cv2.imread(str(args.dataset / "cam0" / f"{i:06d}.jpg"), 0)
        b = cv2.imread(str(args.dataset / "cam0" / f"{i+GAP:06d}.jpg"), 0)
        if a is None or b is None:
            continue
        a2 = cv2.resize(a, None, fx=0.5, fy=0.5)
        b2 = cv2.resize(b, None, fx=0.5, fy=0.5)
        if mask is None:
            mask = np.zeros_like(a2)
            cv2.circle(mask, (int(cx / 2), int(cy / 2)), 660, 255, -1)
        pts = cv2.goodFeaturesToTrack(a2, 250, 0.01, 12, mask=mask)
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
        # translation direction: rows (b1 x R b0)^T t = 0
        Arows = np.cross(b1[idx], (R @ b0[idx].T).T)
        _, _, Vt = np.linalg.svd(Arows)
        tdir_cam = Vt[-1]
        # residual flow after rotation decides sign
        flow = b1[idx] - (R @ b0[idx].T).T
        if np.median(np.einsum("ij,j->i", np.cross(b1[idx], np.cross(tdir_cam[None, :].repeat(len(idx), 0), b1[idx])), np.ones(3) * 0)) == 0:
            pass
        # pick sign that better explains flow: translation toward t makes points flow away from epipole
        s1 = np.median(np.einsum("ij,ij->i", flow, np.cross(np.cross(b1[idx], tdir_cam[None, :].repeat(len(idx), 0)), b1[idx])))
        if s1 < 0:
            tdir_cam = -tdir_cam
        # camera-frame motion -> world: Rw is R_c0(t0)->c0(ti)... maintain world = first cam frame
        t_mid = (i - 1 + GAP / 2) / FPS
        v = np.interp(t_mid, tg, speed)
        p = p + Rw.T @ (tdir_cam * v * (STEP / FPS))
        Rw = R @ Rw
        vo_pts.append(p.copy())
        vo_t.append((i - 1 + GAP) / FPS)
        fwd = Rw.T @ np.array([0, 0, 1.0])
        vo_yaw.append((vo_t[-1], yaw_of(fwd)))

    # ---- gyro-orientation path with GT speed ----
    Rg = np.eye(3)
    p = np.zeros(3)
    gy_pts, gy_t, gy_yaw = [p.copy()], [0.0], []
    k0 = np.searchsorted(ti, 0.0)
    for k in range(k0, len(ti) - 1):
        dt = ti[k + 1] - ti[k]
        th = gyr[k] * dt
        a_ = np.linalg.norm(th)
        if a_ > 1e-12:
            kx = th / a_
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            Rg = Rg @ (np.eye(3) + np.sin(a_) * K + (1 - np.cos(a_)) * K @ K)
        if k % 20 == 0:
            fwd_I = Rg @ f_I  # forward in imu(t0) frame
            v = np.interp(ti[k], tg, speed)
            p = p + fwd_I * v * (20.0 / 997.6)
            gy_pts.append(p.copy())
            gy_t.append(ti[k])
            gy_yaw.append((ti[k], yaw_of(fwd_I)))

    # gravity-align the gyro chain (imu(0) frame): accel-up at start -> +z
    up_body = acc[ti < 1.0].mean(0)
    Ra = rot_a_to_b(up_body, np.array([0, 0, 1.0]))
    gy_pts = [Ra @ p_ for p_ in gy_pts]
    gy_yaw = [(t_, yaw_of(Ra @ np.array([np.cos(y_), np.sin(y_), 0]))) for t_, y_ in gy_yaw] if False else gy_yaw
    # recompute gyro yaws properly in aligned frame
    gy_yaw = []
    Rg2 = np.eye(3)
    for k in range(k0, len(ti) - 1, 20):
        pass
    # simpler: recompute from stored path headings
    gy_yaw = [(gy_t[i], yaw_of(np.array(gy_pts[i]) - np.array(gy_pts[i - 1]))) for i in range(1, len(gy_pts))
              if np.linalg.norm(np.array(gy_pts[i])[:2] - np.array(gy_pts[i - 1])[:2]) > 1e-4]

    # optical chain: cam0-up (-y) -> +z
    Ro = rot_a_to_b(np.array([0, -1.0, 0]), np.array([0, 0, 1.0]))
    vo_pts = [Ro @ p_ for p_ in vo_pts]
    vo_yaw = [(vo_t[i], yaw_of(np.array(vo_pts[i]) - np.array(vo_pts[i - 1]))) for i in range(1, len(vo_pts))
              if np.linalg.norm(np.array(vo_pts[i])[:2] - np.array(vo_pts[i - 1])[:2]) > 1e-4]

    # ---- SLAM yaw ----
    rots = Rotation.from_quat(Q)
    g_world = np.array([0, 0, 9.81])
    am = acc[np.searchsorted(ti, ts[len(ts) // 2])]
    Rm = rots[len(ts) // 2].as_matrix()
    flip = np.linalg.norm(Rm @ am - g_world) < np.linalg.norm(Rm.T @ am - g_world)
    slam_yaw = []
    for k in range(0, len(ts), 5):
        R = rots[k].as_matrix() if flip else rots[k].as_matrix().T
        slam_yaw.append((ts[k], yaw_of(R @ f_I)))

    # ---- GT walk yaw ----
    gt_yaw = [(tg[k], yaw_of(Vg[k])) for k in range(len(tg)) if speed[k] > 0.15]

    # ---- log ----
    bp = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="trajectories", contents="/world/**"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/plots/yaw", name="yaw [deg] per chain", contents="/plots/yaw/**"),
                rrb.TextDocumentView(origin="/legend", name="legend"),
                row_shares=[3, 1],
            ),
            column_shares=[2, 1],
        )
    )
    rr.init("conventions_debug", spawn=False)
    rr.save(str(args.out))
    rr.send_blueprint(bp, make_active=True, make_default=True)
    rr.log("legend", rr.TextDocument(
        "green=GT | blue=SLAM | purple=optical VO (shape only) | yellow=gyro+GTspeed | orange=IMU dead-reckon\n"
        "yaw plot: same colors; mirror/inversion of one curve = convention bug in that chain"), static=True)

    def log_traj(name, P, T, color):
        for i in range(1, len(P)):
            set_t(T[i])
            rr.log(f"world/{name}", rr.LineStrips3D([np.array(P[: i + 1])], colors=[color]))

    Pg_a = Pg
    log_traj("gt", Pg_a, tg, [27, 175, 122])
    Ps_a = align_start(Ps, gt0, gtdir)
    log_traj("slam", Ps_a, ts, [42, 120, 214])
    vo_a = align_start(np.array(vo_pts), gt0, gtdir)
    log_traj("optical_vo", vo_a, vo_t, [150, 80, 220])
    gy_a = align_start(np.array(gy_pts), gt0, gtdir)
    log_traj("gyro_gtspeed", gy_a, gy_t, [235, 161, 0])

    # IMU dead-reckon (reuse simple integration, bias-corrected, 40m cap)
    g0 = acc[ti < 1.0].mean(0)
    zw = np.array([0, 0, 1.0])
    gn = g0 / np.linalg.norm(g0)
    ax_ = np.cross(gn, zw)
    s_ = np.linalg.norm(ax_)
    K = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]]) / (s_ + 1e-12)
    ang0 = np.arctan2(s_, float(gn @ zw))
    R = np.eye(3) + np.sin(ang0) * K * s_ + (1 - np.cos(ang0)) * (K * s_) @ (K * s_) if s_ > 1e-9 else np.eye(3)
    ba = acc[ti < 1.0].mean(0) - R.T @ g_world
    p = np.zeros(3)
    v = np.zeros(3)
    im_pts, im_t = [p.copy()], [ti[0]]
    for k in range(len(ti) - 1):
        dt = ti[k + 1] - ti[k]
        a_w = R @ (acc[k] - ba) - g_world
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        th = gyr[k] * dt
        a_ = np.linalg.norm(th)
        if a_ > 1e-12:
            kx = th / a_
            K2 = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            R = R @ (np.eye(3) + np.sin(a_) * K2 + (1 - np.cos(a_)) * K2 @ K2)
        if np.linalg.norm(p) > 40:
            break
        if k % 100 == 0:
            im_pts.append(p.copy())
            im_t.append(ti[k + 1])
    im_a = align_start(np.array(im_pts), gt0, gtdir)
    log_traj("imu_deadreckon", im_a, im_t, [235, 104, 52])

    # yaw curves (unwrapped, zeroed at own start)
    for name, series, color in [("gt", gt_yaw, None), ("slam", slam_yaw, None),
                                ("optical", vo_yaw, None), ("gyro", gy_yaw, None)]:
        if not series:
            continue
        tt = np.array([x[0] for x in series])
        yy = np.degrees(np.unwrap(np.array([x[1] for x in series])))
        yy -= yy[0]
        for t_, y_ in zip(tt[::2], yy[::2]):
            set_t(float(t_))
            rr.log(f"plots/yaw/{name}", rr.Scalars(float(y_)))
    print(f"saved {args.out}: vo pts={len(vo_pts)}, gyro pts={len(gy_pts)}")


if __name__ == "__main__":
    main()
