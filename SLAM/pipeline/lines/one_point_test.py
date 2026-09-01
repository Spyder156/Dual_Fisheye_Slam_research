#!/usr/bin/env python
"""ONE ORB point, three frames, three cameras, one 3D point. Nothing else.

The math is lifted verbatim from the rerun builder (make_run_output.py), which
we know produces coherent maps:

  pose chain    Twb (trajectory quat+pos) -> Twc = Twb @ T_b_c1 -> Tcw = inv
  unprojection  KB4 Newton solve, pixel -> unit bearing
  projection    KB4 forward, camera point -> pixel

One ORB feature is matched A->B->C, triangulated least-squares from the three
world rays, drawn with the three cameras, and reprojected into all three
images. If this is consistent, the pose/projection chain is proven on the
smallest possible case.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation


def load_cam(cfg, cam=0):
    txt = Path(cfg).read_text()
    def g(k):
        return float(re.search(rf"^{re.escape(k)}:\s*([-\d.eE+]+)", txt, re.M).group(1))
    c = cam + 1
    return dict(fx=g(f"Camera{c}.fx"), fy=g(f"Camera{c}.fy"),
                cx=g(f"Camera{c}.cx"), cy=g(f"Camera{c}.cy"),
                d=[g(f"Camera{c}.k{i}") for i in (1, 2, 3, 4)],
                w=int(g("Camera.width")), h=int(g("Camera.height")))


def T_bc_of(cfg):
    m = re.search(r"IMU.T_b_c1:.*?data:\s*\[(.*?)\]", Path(cfg).read_text(), re.S)
    v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v).reshape(4, 4)


def unproj(u, v, c, it=15):
    """pixel -> unit bearing (KB4). Same Newton solve as the rerun builder."""
    mx = (float(u) - c["cx"]) / c["fx"]
    my = (float(v) - c["cy"]) / c["fy"]
    ru = np.hypot(mx, my)
    k1, k2, k3, k4 = c["d"]
    th = min(ru, np.pi * 0.6)
    for _ in range(it):
        t2 = th * th
        f = th * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4) - ru
        df = 1 + 3*k1*t2 + 5*k2*t2**2 + 7*k3*t2**3 + 9*k4*t2**4
        th = np.clip(th - f / (df if abs(df) > 1e-6 else 1e-6), 0, np.pi * 0.6)
    s = np.sin(th) / ru if ru > 1e-9 else 1.0
    b = np.array([mx * s, my * s, np.cos(th)])
    return b / np.linalg.norm(b)


def proj(Pc, c):
    """camera point -> pixel (KB4 forward). Same as the rerun builder."""
    x, y, z = Pc
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    k1, k2, k3, k4 = c["d"]
    td = th * (1 + k1*th**2 + k2*th**4 + k3*th**6 + k4*th**8)
    s = td / r if r > 1e-9 else 0.0
    return np.array([c["fx"]*x*s + c["cx"], c["fy"]*y*s + c["cy"]]), th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--gap", type=int, default=60)
    ap.add_argument("--gap2", type=int, default=60)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pick", type=int, default=0,
                    help="which 3-frame track (0 = strongest match)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = Path(a.dataset); cam = load_cam(a.config, a.cam); T_bc = T_bc_of(a.config)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    tof = dict(zip(np.atleast_1d(fr["frame"]).astype(int), np.atleast_1d(fr["t"])))
    dtj = np.loadtxt(a.traj); T = dtj[:, 0] / 1e9

    def pose(f):
        """EXACTLY the rerun builder's chain: Twb -> Twc = Twb @ T_bc -> Tcw."""
        j = int(np.argmin(np.abs(T - tof[f])))
        if abs(T[j] - tof[f]) > 0.05:
            raise SystemExit(f"frame {f} has no pose")
        Twb = np.eye(4)
        Twb[:3, :3] = Rotation.from_quat(dtj[j, 4:8]).as_matrix()
        Twb[:3, 3] = dtj[j, 1:4]
        Tcw = np.linalg.inv(Twb @ T_bc)
        return Tcw[:3, :3], Tcw[:3, 3]

    F = [a.frame, a.frame + a.gap, a.frame + a.gap + a.gap2]
    POSE = [pose(f) for f in F]
    CC = [-R.T @ t for (R, t) in POSE]
    IM = [cv2.imread(str(ds / f"cam{a.cam}" / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
          for f in F]
    print(f"frames {F}   baselines {np.linalg.norm(CC[1]-CC[0]):.2f} m, "
          f"{np.linalg.norm(CC[2]-CC[1]):.2f} m")

    # ---- one ORB track through all three frames -----------------------------
    orb = cv2.ORB_create(4000)
    K, D = zip(*[orb.detectAndCompute(im, None) for im in IM])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    mAB = {m.queryIdx: (m.trainIdx, m.distance) for m in bf.match(D[0], D[1])}
    mBC = {m.queryIdx: (m.trainIdx, m.distance) for m in bf.match(D[1], D[2])}
    tracks = []
    for qa, (ib, d1) in mAB.items():
        if ib in mBC:
            ic, d2 = mBC[ib]
            tracks.append((d1 + d2, qa, ib, ic))
    tracks.sort()
    if not tracks:
        raise SystemExit("no ORB track through all three frames")
    print(f"{len(tracks)} ORB tracks persist; using #{a.pick}")
    _, qa, ib, ic = tracks[min(a.pick, len(tracks) - 1)]
    UV = [np.array(K[0][qa].pt), np.array(K[1][ib].pt), np.array(K[2][ic].pt)]

    # ---- bearings and world rays -------------------------------------------
    B = [unproj(uv[0], uv[1], cam) for uv in UV]
    RAYS = [(CC[k], POSE[k][0].T @ B[k]) for k in range(3)]

    # ---- least-squares 3D point from the three rays -------------------------
    A = np.zeros((3, 3)); c = np.zeros(3)
    for C0, r in RAYS:
        M = np.eye(3) - np.outer(r, r)
        A += M; c += M @ C0
    X = np.linalg.solve(A, c)

    # ---- report: reprojection into all three frames -------------------------
    print(f"\n3D point X = {np.round(X, 3)}")
    for k in range(3):
        R, t = POSE[k]
        Pc = R @ X + t
        uv, th = proj(Pc, cam)
        e = np.linalg.norm(uv - UV[k])
        miss = np.linalg.norm(np.cross(X - CC[k], RAYS[k][1]))
        print(f"  f{F[k]}: depth {np.linalg.norm(Pc):6.2f} m  theta {np.degrees(th):5.1f} deg"
              f"  reproj err {e:7.2f} px   ray miss {miss:6.3f} m")

    # ---- rerun --------------------------------------------------------------
    S = float(max(np.linalg.norm(X - c0) for c0 in CC))
    rr.init("one_point", spawn=False); rr.save(a.out)
    rr.set_time("t", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    COL = [[90, 170, 255], [255, 170, 90], [200, 110, 255]]
    for k in range(3):
        R, t = POSE[k]
        rr.log(f"world/cam_f{F[k]}", rr.Transform3D(translation=CC[k], mat3x3=R.T))
        rr.log(f"world/cam_f{F[k]}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]], image_plane_distance=0.05 * S))
        # ray drawn CENTRE -> POINT (slightly past it), so convergence is
        # visible as three lines meeting at the green dot
        rr.log(f"world/ray_f{F[k]}", rr.LineStrips3D(
            [[CC[k], CC[k] + 1.1 * np.linalg.norm(X - CC[k]) * RAYS[k][1]]],
            colors=[COL[k]], radii=0.0015 * S))
    rr.log("world/POINT", rr.Points3D([X], colors=[[60, 230, 90]], radii=0.012 * S))

    for k in range(3):
        v = cv2.cvtColor(IM[k], cv2.COLOR_GRAY2BGR)
        u = UV[k]
        cv2.drawMarker(v, (int(u[0]), int(u[1])), (255, 255, 255),
                       cv2.MARKER_CROSS, 40, 4)                      # observed
        uv, th = proj(POSE[k][0] @ X + POSE[k][1], cam)
        if th < np.deg2rad(95):
            cv2.circle(v, (int(uv[0]), int(uv[1])), 14, (90, 240, 90), 3)  # reprojected
        rr.log(f"frame{k}_f{F[k]}",
               rr.Image(cv2.rotate(v, cv2.ROTATE_180)).compress(jpeg_quality=92))

    print("\nWHITE cross = observed ORB pixel | GREEN circle = reprojected 3D point")
    print("3D: three frusta, three coloured rays, one green point")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
