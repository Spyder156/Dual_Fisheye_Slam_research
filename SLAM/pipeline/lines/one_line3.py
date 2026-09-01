#!/usr/bin/env python
"""ONE ELSED line, three frames: endpoints via POINT math, line via LINE math.

Extends one_point_test.py (verified: 1-5 px reprojection) from a point to a
line. Same pose chain, same KB4 in both directions.

  GREEN points   each endpoint pixel, tracked through the 3 frames, unprojected
                 and triangulated least-squares from the 3 world rays -- the
                 identical math that placed the single ORB point correctly.

  YELLOW line    the 3 great-circle normals (n = b1 x b2 per frame), Plucker
                 line least-squares through the 3 interpretation planes -- the
                 ELSED math.

The hope being tested: the endpoints sit on the line's ends in every image, so
the green points should sit on the yellow line's ends in 3D too.

Endpoint ORDER is aligned across frames by bearing dot product before anything
is triangulated (the matcher is orientation-free, so endpoint 1 in frame A can
be endpoint 2 in frame B).
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

GATE_N, GATE_D, GATE_L = np.deg2rad(3.0), np.deg2rad(8.0), 0.5


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
    mx = (np.asarray(u, float) - c["cx"]) / c["fx"]
    my = (np.asarray(v, float) - c["cy"]) / c["fy"]
    ru = np.hypot(mx, my)
    k1, k2, k3, k4 = c["d"]
    th = np.clip(np.atleast_1d(ru).astype(float), 0, np.pi * 0.6)
    for _ in range(it):
        t2 = th * th
        f = th * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4) - ru
        df = 1 + 3*k1*t2 + 5*k2*t2**2 + 7*k3*t2**3 + 9*k4*t2**4
        th = np.clip(th - f / np.where(np.abs(df) < 1e-6, 1e-6, df), 0, np.pi * 0.6)
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    return b / np.linalg.norm(b, axis=-1, keepdims=True)


def proj(Pc, c):
    Pc = np.atleast_2d(Pc)
    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    k1, k2, k3, k4 = c["d"]
    td = th * (1 + k1*th**2 + k2*th**4 + k3*th**6 + k4*th**8)
    s = np.where(r > 1e-9, td / np.maximum(r, 1e-9), 0.0)
    return np.stack([c["fx"]*x*s + c["cx"], c["fy"]*y*s + c["cy"]], 1), th


def elsed(tool, dd, f, out):
    Path("/tmp/_l3.list").write_text(f"{f}\n")
    subprocess.run([tool, str(dd), "/tmp/_l3.list", out, "30", "15"],
                   capture_output=True, check=True)
    return np.array([[float(x) for x in l.split(",")[1:5]]
                     for l in open(out).read().splitlines()[1:]])


def lift(segs, cam, mn=np.deg2rad(1.5), mx=np.deg2rad(40.0)):
    b1 = unproj(segs[:, 0], segs[:, 1], cam)
    b2 = unproj(segs[:, 2], segs[:, 3], cam)
    n = np.cross(b1, b2)
    ln = np.linalg.norm(n, axis=1)
    ang = np.arcsin(np.clip(ln, 0, 1))
    k = (ln > 1e-9) & (ang >= mn) & (ang <= mx)
    n = n[k] / ln[k, None]
    d = b2[k] - b1[k]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return segs[k], n, d, ang[k], b1[k], b2[k]


def match(nB, dB, aB, nA, dA, aA):
    an = np.abs(nB @ nA.T); ad = np.abs(dB @ dA.T)
    lo = np.minimum(aB[:, None], aA[None, :]); hi = np.maximum(aB[:, None], aA[None, :])
    ok = (an >= np.cos(GATE_N)) & (ad >= np.cos(GATE_D)) & (lo/np.maximum(hi, 1e-12) >= GATE_L)
    cand = sorted(((an[i, j], i, j) for i, j in zip(*np.where(ok))), key=lambda c: -c[0])
    asg = np.full(len(nB), -1, int); used = np.zeros(len(nA), bool)
    for _, i, j in cand:
        if asg[i] == -1 and not used[j]:
            asg[i] = j; used[j] = True
    return asg


def tri_point(bs, poses):
    """Least-squares point from 3 rays -- identical to one_point_test."""
    A = np.zeros((3, 3)); c = np.zeros(3)
    for b, (R, t) in zip(bs, poses):
        r = R.T @ b; C = -R.T @ t
        M = np.eye(3) - np.outer(r, r)
        A += M; c += M @ C
    return np.linalg.solve(A, c)


def line_from_planes(normals, poses):
    """Least-squares Plucker line through the 3 interpretation planes."""
    N = np.stack([R.T @ n for n, (R, t) in zip(normals, poses)])
    C = np.stack([-R.T @ t for (R, t) in poses])
    _, _, Vt = np.linalg.svd(N)
    d = Vt[-1] / np.linalg.norm(Vt[-1])
    A = np.vstack([N, d[None, :]])
    b = np.concatenate([np.einsum("ij,ij->i", N, C), [0.0]])
    P, *_ = np.linalg.lstsq(A, b, rcond=None)
    return d, np.cross(P, d)


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
    ap.add_argument("--pick", type=int, default=0,
                    help="rank among persisting lines, ordered by how well the "
                         "endpoint rays converge (0 = most point-like corners)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = Path(a.dataset); cam = load_cam(a.config, a.cam); T_bc = T_bc_of(a.config)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    tof = dict(zip(np.atleast_1d(fr["frame"]).astype(int), np.atleast_1d(fr["t"])))
    dtj = np.loadtxt(a.traj); T = dtj[:, 0] / 1e9

    def pose(f):
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
    L = [lift(elsed(a.elsed, ds / f"cam{a.cam}", f, f"/tmp/_m{i}.csv"), cam)
         for i, f in enumerate(F)]
    print(f"frames {F}   baselines {np.linalg.norm(CC[1]-CC[0]):.2f} m, "
          f"{np.linalg.norm(CC[2]-CC[1]):.2f} m")

    a10 = match(L[1][1], L[1][2], L[1][3], L[0][1], L[0][2], L[0][3])
    a21 = match(L[2][1], L[2][2], L[2][3], L[1][1], L[1][2], L[1][3])
    triples = [(a10[i1], i1, i2) for i2, i1 in enumerate(a21)
               if i1 >= 0 and a10[i1] >= 0]
    if not triples:
        raise SystemExit("no line persists through all three frames")

    def endpoints_of(tr):
        """Endpoint bearings per frame, ORDER-ALIGNED to frame A's endpoint 1/2.

        The matcher is orientation-free, so frame B's stored endpoint 1 can be
        frame A's endpoint 2. Align by bearing similarity in the world frame
        (rotations only -- translation moves bearings far less than the
        endpoint separation for a visible segment).
        """
        out = []
        ref = [POSE[0][0].T @ L[0][4][tr[0]], POSE[0][0].T @ L[0][5][tr[0]]]
        for k in range(3):
            w1 = POSE[k][0].T @ L[k][4][tr[k]]
            w2 = POSE[k][0].T @ L[k][5][tr[k]]
            keep = (w1 @ ref[0] + w2 @ ref[1]) >= (w1 @ ref[1] + w2 @ ref[0])
            out.append((L[k][4][tr[k]], L[k][5][tr[k]]) if keep
                       else (L[k][5][tr[k]], L[k][4][tr[k]]))
        return out

    def solve(tr):
        eps = endpoints_of(tr)
        E, miss = [], []
        for e in (0, 1):
            bs = [eps[k][e] for k in range(3)]
            X = tri_point(bs, POSE)
            E.append(X)
            miss.append(max(np.linalg.norm(np.cross(X - CC[k], POSE[k][0].T @ bs[k]))
                            for k in range(3)))
        normals = [L[k][1][tr[k]] for k in range(3)]
        dw, mw = line_from_planes(normals, POSE)
        return E, miss, dw, mw

    scored = []
    for q, tr in enumerate(triples):
        try:
            E, miss, dw, mw = solve(tr)
        except np.linalg.LinAlgError:
            continue
        scored.append((max(miss), tr, E, miss, dw, mw))
    scored.sort(key=lambda z: z[0])

    print(f"\n{len(scored)} lines persist. ranked by worst endpoint-ray miss:")
    print(f"{'rank':>5s}{'angLen':>9s}{'miss e1':>10s}{'miss e2':>10s}"
          f"{'e1 off line':>13s}{'e2 off line':>13s}")
    for rank, (mx, tr, E, miss, dw, mw) in enumerate(scored[:10]):
        p0 = np.cross(dw, mw)
        offs = [np.linalg.norm((X - p0) - dw * ((X - p0) @ dw)) for X in E]
        print(f"{rank:5d}{np.degrees(L[0][3][tr[0]]):8.2f}d{miss[0]:10.3f}{miss[1]:10.3f}"
              f"{offs[0]:13.3f}{offs[1]:13.3f}")

    mx, tr, E, miss, dw, mw = scored[min(a.pick, len(scored) - 1)]
    p0 = np.cross(dw, mw)
    feet = [p0 + dw * ((X - p0) @ dw) for X in E]
    offs = [np.linalg.norm(X - f0) for X, f0 in zip(E, feet)]

    # THE LINE'S OWN EXTENT -- SetExtentFromBearings, the SLAM's recipe:
    # intersect each observed endpoint bearing of frame A with the line,
    #   s = (m_c . (b x d_c)) / |b x d_c|^2
    # This is drawn AS RETURNED. The previous version drew the segment between
    # the perpendicular feet of the green points instead, which guaranteed the
    # yellow line spanned the green dots -- agreement by construction, not
    # measurement. Never again.
    epsA = endpoints_of(tr)[0]
    R0_, t0_ = POSE[0]
    d_c = R0_ @ dw; m_c = R0_ @ mw + np.cross(t0_, d_c)
    yellow, svals = [], []
    for b in epsA:
        cr = np.cross(b, d_c)
        sv = (m_c @ cr) / max(cr @ cr, 1e-15)
        svals.append(sv)
        yellow.append(R0_.T @ (sv * b - t0_))
    print(f"\nusing rank {a.pick}:")
    print(f"  GREEN endpoint1 {np.round(E[0],3)}   miss {miss[0]:.3f} m   "
          f"off the yellow line {offs[0]:.3f} m")
    print(f"  GREEN endpoint2 {np.round(E[1],3)}   miss {miss[1]:.3f} m   "
          f"off the yellow line {offs[1]:.3f} m")
    print(f"  YELLOW segment via the ELSED extent math (s = intersection depth):")
    print(f"    s = {svals[0]:.3f} m, {svals[1]:.3f} m"
          + ("   <- NEGATIVE: behind the camera" if min(svals) < 0 else ""))
    print(f"    {np.round(yellow[0],3)}   {np.round(yellow[1],3)}")
    print(f"  feet of the green points on the same line (for comparison only):")
    print(f"    {np.round(feet[0],3)}   {np.round(feet[1],3)}")
    for k in range(3):
        Rk, tk = POSE[k]
        nline = Rk @ mw + np.cross(tk, Rk @ dw)
        nline /= np.linalg.norm(nline)
        rd = [np.degrees(np.arcsin(np.clip(abs(nline @ b), 0, 1)))
              for b in endpoints_of(tr)[k]]
        print(f"  f{F[k]}: line residual on observed bearings "
              f"{rd[0]:.3f} / {rd[1]:.3f} deg")

    # ------------------------------------------------------------------ viz --
    mid = 0.5 * (yellow[0] + yellow[1])
    S = float(max(np.linalg.norm(mid - c0) for c0 in CC))
    rr.init("one_line3", spawn=False); rr.save(a.out)
    rr.set_time("t", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    COL = [[90, 170, 255], [255, 170, 90], [200, 110, 255]]
    eps = endpoints_of(tr)
    for k in range(3):
        Rk, tk = POSE[k]
        rr.log(f"world/cam_f{F[k]}", rr.Transform3D(translation=CC[k], mat3x3=Rk.T))
        rr.log(f"world/cam_f{F[k]}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]], image_plane_distance=0.05 * S))
        for e in (0, 1):
            r = Rk.T @ eps[k][e]
            L_r = 1.1 * np.linalg.norm(E[e] - CC[k])
            rr.log(f"world/ray_f{F[k]}_e{e+1}", rr.LineStrips3D(
                [[CC[k], CC[k] + L_r * r]], colors=[COL[k]], radii=0.0012 * S))
    rr.log("world/LINE_math", rr.LineStrips3D(
        [[yellow[0], yellow[1]]], colors=[[245, 210, 50]], radii=0.006 * S))
    rr.log("world/POINT_math", rr.Points3D(
        np.stack(E), colors=[[60, 230, 90]] * 2, radii=0.011 * S))

    for k in range(3):
        Rk, tk = POSE[k]
        v = cv2.cvtColor(IM[k], cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = L[k][0][tr[k]]
        cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 6, cv2.LINE_AA)
        samp = np.stack([yellow[0] + s * (yellow[1] - yellow[0])
                         for s in np.linspace(-0.25, 1.25, 60)])
        uv, th = proj((Rk @ samp.T).T + tk, cam)
        pts = uv[th < np.deg2rad(95)].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(v, [pts.reshape(-1, 1, 2)], False, (50, 210, 245), 3, cv2.LINE_AA)
        for X in E:
            p, thp = proj((Rk @ X + tk)[None, :], cam)
            if thp[0] < np.deg2rad(95):
                cv2.circle(v, (int(p[0, 0]), int(p[0, 1])), 13, (80, 240, 90), 3)
        rr.log(f"frame{k}_f{F[k]}",
               rr.Image(cv2.rotate(v, cv2.ROTATE_180)).compress(jpeg_quality=92))

    print("\nWHITE = detected segment | YELLOW = LINE-math line | GREEN = POINT-math endpoints")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
