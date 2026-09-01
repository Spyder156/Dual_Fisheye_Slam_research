#!/usr/bin/env python
"""Image -> 3D, step by step, VISUALIZED: an ELSED endpoint next to an ORB corner.

Same three frames, same poses, same KB4 unprojection code. Two pixel tracks:

  MAGENTA   an ORB corner tracked through the 3 frames (picked to be spatially
            CLOSE to the line in frame A) -- the case that is proven to work.
  YELLOW    endpoint 1 of the matched ELSED line in the same 3 frames.

For each, the three world rays are drawn from the camera centres. Where the
rays converge (or fail to) IS the image->3D math, visible.

Per pixel, the round trip  pixel -> bearing -> pixel  is printed, so any error
in the unprojection itself (as opposed to the pixels meaning different physical
points) is measured separately.
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from one_line3 import (load_cam, T_bc_of, unproj, proj, elsed, lift, match,
                       tri_point, GATE_N, GATE_D, GATE_L)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--elsed", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--gap", type=int, default=60)
    ap.add_argument("--gap2", type=int, default=60)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = Path(a.dataset); cam = load_cam(a.config, a.cam); T_bc = T_bc_of(a.config)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    tof = dict(zip(np.atleast_1d(fr["frame"]).astype(int), np.atleast_1d(fr["t"])))
    dtj = np.loadtxt(a.traj); T = dtj[:, 0] / 1e9

    def pose(f):
        j = int(np.argmin(np.abs(T - tof[f])))
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

    # ---- the matched ELSED line, same ranking as one_line3 ------------------
    L = [lift(elsed(a.elsed, ds / f"cam{a.cam}", f, f"/tmp/_r{i}.csv"), cam)
         for i, f in enumerate(F)]
    a10 = match(L[1][1], L[1][2], L[1][3], L[0][1], L[0][2], L[0][3])
    a21 = match(L[2][1], L[2][2], L[2][3], L[1][1], L[1][2], L[1][3])
    triples = [(a10[i1], i1, i2) for i2, i1 in enumerate(a21)
               if i1 >= 0 and a10[i1] >= 0]

    def endpoints_of(tr):
        ref = [POSE[0][0].T @ L[0][4][tr[0]], POSE[0][0].T @ L[0][5][tr[0]]]
        out = []
        for k in range(3):
            w1 = POSE[k][0].T @ L[k][4][tr[k]]
            w2 = POSE[k][0].T @ L[k][5][tr[k]]
            keep = (w1 @ ref[0] + w2 @ ref[1]) >= (w1 @ ref[1] + w2 @ ref[0])
            out.append((L[k][4][tr[k]], L[k][5][tr[k]]) if keep
                       else (L[k][5][tr[k]], L[k][4][tr[k]]))
        return out

    def rank_key(tr):
        eps = endpoints_of(tr)
        worst = 0.0
        for e in (0, 1):
            X = tri_point([eps[k][e] for k in range(3)], POSE)
            worst = max(worst, max(np.linalg.norm(np.cross(X - CC[k],
                        POSE[k][0].T @ eps[k][e])) for k in range(3)))
        return worst

    triples.sort(key=rank_key)
    tr = triples[min(a.pick, len(triples) - 1)]
    eps = endpoints_of(tr)
    # the endpoint PIXELS (recover from the stored segments, order-aligned)
    def px_of(k, e):
        x1, y1, x2, y2 = L[k][0][tr[k]]
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        # which stored pixel corresponds to the order-aligned bearing e?
        b1 = L[k][4][tr[k]]
        return (p1 if np.allclose(eps[k][e], b1) else p2) if e == 0 else \
               (p2 if np.allclose(eps[k][0], b1) else p1)
    LUV = [[px_of(k, e) for e in (0, 1)] for k in range(3)]

    # ---- an ORB corner track, nearest to the line in frame A ----------------
    orb = cv2.ORB_create(4000)
    K, D = zip(*[orb.detectAndCompute(im, None) for im in IM])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    mAB = {m.queryIdx: (m.trainIdx, m.distance) for m in bf.match(D[0], D[1])}
    mBC = {m.queryIdx: (m.trainIdx, m.distance) for m in bf.match(D[1], D[2])}
    tracks = [(qa, ib, mBC[ib][0], d1 + mBC[ib][1])
              for qa, (ib, d1) in mAB.items() if ib in mBC]
    anchor = LUV[0][0]
    # nearest-to-anchor alone picked a FALSE match (pixel jumped 1128->346->978
    # across the frames). Require appearance quality first, then take the
    # closest of the good ones.
    good = sorted(tracks, key=lambda t: t[3])[:len(tracks) // 4 or 1]
    good.sort(key=lambda t: np.linalg.norm(np.array(K[0][t[0]].pt) - anchor))
    # the reference must PROVE it is one physical corner: triangulate and
    # require < 8 px reprojection in all three frames, else try the next.
    OUV = None
    for qa, ib, ic, _ in good:
        cand = [np.array(K[0][qa].pt), np.array(K[1][ib].pt), np.array(K[2][ic].pt)]
        Xc = tri_point([unproj(*u, cam).reshape(-1) for u in cand], POSE)
        ok = True
        for k in range(3):
            Pc = POSE[k][0] @ Xc + POSE[k][1]
            uv2, th2 = proj(Pc[None, :], cam)
            if th2[0] > np.deg2rad(95) or np.linalg.norm(uv2[0] - cand[k]) > 8.0:
                ok = False; break
        if ok:
            OUV = cand; break
    if OUV is None:
        raise SystemExit("no verified ORB track near the line")
    print(f"frames {F}; ORB corner {np.linalg.norm(OUV[0]-anchor):.0f} px from "
          f"the line endpoint in frame A\n")

    # ---- image -> 3D for both, with the round trip printed ------------------
    def ray(uv, k):
        b = unproj(uv[0], uv[1], cam)
        b = b[0] if b.ndim == 2 else b
        uv2, th = proj((b * 1.0)[None, :], cam)
        rt = float(np.linalg.norm(uv2[0] - uv))
        return b, POSE[k][0].T @ b, np.degrees(np.arccos(np.clip(b[2], -1, 1))), rt

    print(f"{'':10s}{'pixel':>18s}{'theta':>8s}{'roundtrip':>10s}")
    O_rays, L_rays = [], []
    for k in range(3):
        b, rw, th, rt = ray(OUV[k], k)
        O_rays.append(rw)
        print(f"ORB   f{F[k]}{str(np.round(OUV[k],0)):>18s}{th:7.1f}d{rt:9.3f}px")
    for k in range(3):
        b, rw, th, rt = ray(LUV[k][0], k)
        L_rays.append(rw)
        print(f"LINE  f{F[k]}{str(np.round(LUV[k][0],0)):>18s}{th:7.1f}d{rt:9.3f}px")

    XO = tri_point([unproj(*OUV[k], cam).reshape(-1) for k in range(3)], POSE)
    XL = tri_point([eps[k][0] for k in range(3)], POSE)
    print(f"\nORB  corner triangulates to {np.round(XO,3)}")
    for k in range(3):
        zc = (POSE[k][0] @ XO + POSE[k][1])[2]
        miss = np.linalg.norm(np.cross(XO - CC[k], O_rays[k]))
        print(f"   f{F[k]}: z_cam {zc:8.3f}   ray miss {miss:.3f} m")
    print(f"LINE endpoint1 triangulates to {np.round(XL,3)}")
    for k in range(3):
        zc = (POSE[k][0] @ XL + POSE[k][1])[2]
        miss = np.linalg.norm(np.cross(XL - CC[k], L_rays[k]))
        print(f"   f{F[k]}: z_cam {zc:8.3f}   ray miss {miss:.3f} m")

    # ---- rerun --------------------------------------------------------------
    S = float(max(np.linalg.norm(XO - c0) for c0 in CC))
    rr.init("rays_side_by_side", spawn=False); rr.save(a.out)
    rr.set_time("t", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for k in range(3):
        Rk, tk = POSE[k]
        rr.log(f"world/cam_f{F[k]}", rr.Transform3D(translation=CC[k], mat3x3=Rk.T))
        rr.log(f"world/cam_f{F[k]}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]], image_plane_distance=0.05 * S))
        rr.log(f"world/ORB_ray_f{F[k]}", rr.LineStrips3D(
            [[CC[k], CC[k] + 1.15 * np.linalg.norm(XO - CC[k]) * O_rays[k]]],
            colors=[[255, 60, 255]], radii=0.0015 * S))
        rr.log(f"world/LINE_ray_f{F[k]}", rr.LineStrips3D(
            [[CC[k], CC[k] + 1.15 * max(np.linalg.norm(XL - CC[k]), 0.4 * S) * L_rays[k]]],
            colors=[[245, 210, 50]], radii=0.0015 * S))
    rr.log("world/ORB_point", rr.Points3D([XO], colors=[[255, 60, 255]], radii=0.012 * S))
    rr.log("world/LINE_endpoint1", rr.Points3D([XL], colors=[[245, 210, 50]], radii=0.012 * S))

    for k in range(3):
        v = cv2.cvtColor(IM[k], cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = L[k][0][tr[k]]
        cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 4, cv2.LINE_AA)
        u = LUV[k][0]
        cv2.circle(v, (int(u[0]), int(u[1])), 14, (50, 210, 245), 3)      # line endpoint
        o = OUV[k]
        cv2.drawMarker(v, (int(o[0]), int(o[1])), (255, 60, 255),
                       cv2.MARKER_CROSS, 36, 4)                           # ORB corner
        rr.log(f"frame{k}_f{F[k]}",
               rr.Image(cv2.rotate(v, cv2.ROTATE_180)).compress(jpeg_quality=92))

    print("\nMAGENTA = ORB corner (cross in images, rays+dot in 3D)")
    print("YELLOW  = line endpoint 1 (circle in images, rays+dot in 3D)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
