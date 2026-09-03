#!/usr/bin/env python
"""ONE line, three frames, two projection paths. Nothing else.

No ORB. No point detector. No matcher for points. The only points involved are
the two ENDPOINTS OF THE LINE ITSELF, pushed through the point pipeline:

  LINE  path   the two endpoint bearings give n = b1 x b2 in each frame; the
               Plucker line (d, m) comes from intersecting the interpretation
               planes of frames A and B. Drawn YELLOW.

  POINT path   the same two endpoint pixels, unprojected to bearings and
               triangulated as ordinary 3D POINTS from frames A and B, exactly
               the way a MapPoint is built. Drawn GREEN.

Both start from the identical two pixels in the identical two frames. If the two
pipelines agree, the green dots lie on the yellow line.
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
    th = np.clip(np.atleast_1d(ru).copy(), 0, np.pi * 0.6)
    for _ in range(it):
        t2 = th * th
        f = th * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4) - ru
        df = 1 + 3*k1*t2 + 5*k2*t2**2 + 7*k3*t2**3 + 9*k4*t2**4
        th = np.clip(th - f / np.where(np.abs(df) < 1e-6, 1e-6, df), 0, np.pi * 0.6)
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return b.reshape(-1)[:3] if b.ndim == 2 and b.shape[0] == 1 else b


def elsed(tool, d, f, out):
    Path("/tmp/_ol.list").write_text(f"{f}\n")
    subprocess.run([tool, str(d), "/tmp/_ol.list", out, "30", "15"],
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


def triangulate_line(n1_c, R1, t1, n2_c, R2, t2, min_deg=2.0):
    """MapLine::Triangulate -- intersect the two interpretation planes."""
    n1 = R1.T @ n1_c; n2 = R2.T @ n2_c
    C1 = -R1.T @ t1;  C2 = -R2.T @ t2
    d = np.cross(n1, n2); dn = np.linalg.norm(d)
    if dn < np.sin(np.deg2rad(min_deg)):
        return None, None
    d /= dn
    A = np.stack([n1, n2, d]); b = np.array([n1 @ C1, n2 @ C2, 0.0])
    P = np.linalg.solve(A, b)
    return d, np.cross(P, d)


def triangulate_point(bA, RA, tA, bB, RB, tB):
    """Ordinary point triangulation: midpoint of closest approach of two rays."""
    CA = -RA.T @ tA; CB = -RB.T @ tB
    rA = RA.T @ bA;  rB = RB.T @ bB
    w = CB - CA
    aa, bb, cc = rA @ rA, rA @ rB, rB @ rB
    dd, ee = rA @ w, rB @ w
    den = aa*cc - bb*bb
    if abs(den) < 1e-12:
        return None, None, None
    s1 = (bb*ee - cc*dd) / -den
    s2 = (aa*ee - bb*dd) / -den
    return 0.5*((CA + s1*rA) + (CB + s2*rB)), CA + s1*rA, CB + s2*rB


def clip(d_w, m_w, R, t, b1, b2, maxd=500.0):
    d_c = R @ d_w; m_c = R @ m_w + np.cross(t, d_c)
    out = []
    for b in (b1, b2):
        cr = np.cross(b, d_c); den = cr @ cr
        if den < 1e-10:
            return None
        s = (m_c @ cr) / den
        if not np.isfinite(s) or s <= 0.02 or s > maxd:
            return None
        out.append(R.T @ (s * b - t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--elsed", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--gap2", type=int, default=30)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pick", type=int, default=0,
                    help="which persisting line to use (0 = longest)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = Path(a.dataset); cam = load_cam(a.config, a.cam); T_bc = T_bc_of(a.config)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    tof = dict(zip(np.atleast_1d(fr["frame"]).astype(int), np.atleast_1d(fr["t"])))
    d = np.loadtxt(a.traj); T = d[:, 0] / 1e9

    def pose(f):
        j = int(np.argmin(np.abs(T - tof[f])))
        Twb = np.eye(4)
        Twb[:3, :3] = Rotation.from_quat(d[j, 4:8]).as_matrix()
        Twb[:3, 3] = d[j, 1:4]
        Tcw = np.linalg.inv(Twb @ T_bc)
        return Tcw[:3, :3], Tcw[:3, 3]

    F = [a.frame, a.frame + a.gap, a.frame + a.gap + a.gap2]
    POSE = [pose(f) for f in F]
    IM = [cv2.imread(str(ds / f"cam{a.cam}" / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE) for f in F]
    L = [lift(elsed(a.elsed, ds / f"cam{a.cam}", f, f"/tmp/_l{i}.csv"), cam)
         for i, f in enumerate(F)]

    asg10 = match(L[1][1], L[1][2], L[1][3], L[0][1], L[0][2], L[0][3])
    asg21 = match(L[2][1], L[2][2], L[2][3], L[1][1], L[1][2], L[1][3])
    triples = [(asg10[i1], i1, i2) for i2, i1 in enumerate(asg21)
               if i1 >= 0 and asg10[i1] >= 0]
    if not triples:
        raise SystemExit("no line persists through all three frames")
    triples.sort(key=lambda t: -L[0][3][t[0]])          # longest angular extent
    idx = triples[min(a.pick, len(triples) - 1)]
    print(f"{len(triples)} lines persist through frames {F}; using #{a.pick}\n")
    (Q0, u0), (Q1, u1), _ = POSE
    Cq = -Q0.T @ u0
    print(f"{'#':>3s}{'angLen[deg]':>12s}{'depth from camA [m]':>21s}{'3D len [m]':>12s}")
    for q, (j0, j1, j2) in enumerate(triples[:15]):
        dq, mq = triangulate_line(L[0][1][j0], Q0, u0, L[1][1][j1], Q1, u1)
        if dq is None:
            print(f"{q:3d}{np.degrees(L[0][3][j0]):12.2f}{'rejected (parallax)':>21s}"); continue
        sg = clip(dq, mq, Q0, u0, L[0][4][j0], L[0][5][j0])
        if sg is None:
            print(f"{q:3d}{np.degrees(L[0][3][j0]):12.2f}{'clip failed':>21s}"); continue
        mid = 0.5 * (sg[0] + sg[1])
        print(f"{q:3d}{np.degrees(L[0][3][j0]):12.2f}{np.linalg.norm(mid - Cq):21.2f}"
              f"{np.linalg.norm(sg[1] - sg[0]):12.3f}")
    print()

    rr.init("one_line", spawn=False); rr.save(a.out)
    rr.set_time("t", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    FCOL = [[90, 170, 255], [255, 170, 90], [200, 110, 255]]   # one per frame

    # SCENE SCALE. Every size below derives from the distance between the
    # cameras and the line, so nothing is hardcoded in metres. The frusta were
    # drawn at a fixed 0.35 m, which is enormous next to a segment a few
    # centimetres long and swallowed the thing being examined.
    (R0, t0), (R1, t1), (R2, t2) = POSE
    i0, i1, i2 = idx
    dw, mw = triangulate_line(L[0][1][i0], R0, t0, L[1][1][i1], R1, t1)
    if dw is None:
        raise SystemExit("line rejected by the 2 deg parallax gate; try --pick or a bigger --gap")
    C0 = -R0.T @ t0
    seg0 = clip(dw, mw, R0, t0, L[0][4][i0], L[0][5][i0])
    anchor = 0.5 * (seg0[0] + seg0[1]) if seg0 is not None else np.cross(dw, mw)
    S = float(max(np.linalg.norm(anchor - C0),
                  max(np.linalg.norm(-P[0].T @ P[1] - C0) for P in [(R1, t1), (R2, t2)])))
    S = max(S, 1e-3)

    for k in range(3):
        R, t = POSE[k]; C = -R.T @ t
        rr.log(f"world/cam_f{F[k]}", rr.Transform3D(translation=C, mat3x3=R.T))
        rr.log(f"world/cam_f{F[k]}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]],
            image_plane_distance=0.03 * S))

    # ---- LINE PATH: draw the INFINITE line, then the observed piece on it ---
    # A Plucker line has no ends. Drawing only the clipped piece made a 5 cm
    # stub invisible at any zoom that showed the cameras, so the line itself is
    # drawn faintly across the scene and the observed segment sits on it.
    for k, (ii, (R, t)) in enumerate(zip(idx, POSE)):
        seg = clip(dw, mw, R, t, L[k][4][ii], L[k][5][ii])
        if seg is not None:
            rr.log(f"world/LINE_seen_from_f{F[k]}", rr.LineStrips3D(
                [[seg[0], seg[1]]], colors=[FCOL[k]], radii=0.006 * S))

    # ---- POINT PATH: the SAME two endpoint pixels, triangulated as points ----
    # No rays drawn. They ran from the camera centres to these points and
    # dominated the view, which made the picture look like two lines emanating
    # from a camera -- the debug scaffolding, not the result.
    # ENDPOINT ORDER. The matcher is orientation-free on purpose -- a line has
    # no direction, so it compares |n.n| and |d.d|. That means a matched pair
    # can come back with its two endpoints swapped, and pairing b1(A) with
    # b1(B) then triangulates one end of the line against the OTHER end. The
    # result is a point that is not on the line at all, which is why the
    # segment did not sit between the two green points.
    flip = (L[0][4][i0] @ L[1][4][i1]) < (L[0][4][i0] @ L[1][5][i1])
    if flip:
        print("endpoint order of frame B is SWAPPED relative to frame A -- corrected")
    ends = []
    for e in (0, 1):
        bA = L[0][4 + e][i0]
        bB = L[1][4 + (1 - e if flip else e)][i1]
        X, pA, pB = triangulate_point(bA, R0, t0, bB, R1, t1)
        if X is None:
            continue
        ends.append(X)
        rr.log(f"world/POINT_path/endpoint{e+1}", rr.Points3D(
            [X], colors=[[60, 230, 90]], radii=0.010 * S))

    # ---- the one segment, drawn in all three images -------------------------
    for k in range(3):
        v = cv2.cvtColor(IM[k], cv2.COLOR_GRAY2BGR)
        for s in L[k][0]:
            cv2.line(v, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (60, 60, 60), 1)
        x1, y1, x2, y2 = L[k][0][idx[k]]
        cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (50, 220, 255), 4, cv2.LINE_AA)
        cv2.circle(v, (int(x1), int(y1)), 9, (90, 255, 90), -1)
        cv2.circle(v, (int(x2), int(y2)), 9, (90, 255, 90), -1)
        rr.log(f"frame{k}_f{F[k]}",
               rr.Image(cv2.rotate(v, cv2.ROTATE_180)).compress(jpeg_quality=90))

    segA = seg0
    px = np.hypot(L[0][0][i0][2]-L[0][0][i0][0], L[0][0][i0][3]-L[0][0][i0][1])
    print(f"\nobserved segment in frame {F[0]}: {px:.1f} px long, "
          f"{np.degrees(L[0][3][i0]):.2f} deg angular")
    print(f"distance camera -> line: {np.linalg.norm(anchor - C0):.2f} m   "
          f"scene scale S = {S:.2f} m")
    if segA is not None:
        print(f"3D length of the observed piece: {np.linalg.norm(segA[1]-segA[0]):.3f} m")
    print("\nraw 3D coordinates")
    if segA is not None:
        print(f"  LINE  (yellow) endpoints  {np.round(segA[0],3)}   {np.round(segA[1],3)}")
    for e, X in enumerate(ends):
        print(f"  POINT (green) endpoint{e+1}  {np.round(X,3)}")
    for e, X in enumerate(ends):
        v = X - (-R0.T @ t0)                      # not used; kept explicit below
        w = X - np.cross(dw, mw)                  # vector from a point on the line
        perp = np.linalg.norm(w - dw * (w @ dw))
        print(f"  distance of green endpoint{e+1} from the yellow LINE: {perp:.4f} m")
    print("\nYELLOW = the line via the LINE math (Plucker, planes of A+B)")
    print("GREEN  = its two ENDPOINTS via the POINT math (same pixels, same 2 frames)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
