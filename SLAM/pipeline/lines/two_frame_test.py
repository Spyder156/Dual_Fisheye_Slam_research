#!/usr/bin/env python
"""THREE real frames: triangulate from two, judge on a third it never saw.

Two frames is not enough to test a line. Triangulation intersects the two
interpretation planes EXACTLY, so the line it returns lies perfectly in both --
the residual in those two frames is identically zero by construction, however
wrong the depth is. Measuring there proves nothing (it printed 0.000 across
every percentile, which is what gave it away).

Points do not have this property: a point from two rays is a least-squares
midpoint, so it keeps a non-zero residual. Comparing the two in the SAME two
frames is therefore not even a fair comparison -- one is exact by construction
and the other is not.

So: triangulate from A and B, then evaluate on a HELD-OUT frame C. Nothing is
fitted to C. Both lines and points are judged the same way, on data neither was
given.

The previous version of this test was circular -- it built the Plucker line out
of the very 3D points it then "recovered", so the answer was an algebraic
identity and could not fail. Nothing here is allowed to see a depth it did not
earn:

  LINES   observation A alone gives n1 = b1 x b2.  Observation B alone gives n2.
          Both are pure image measurements. (d, m) comes from intersecting the
          two interpretation planes using the two REAL poses. The result is then
          projected back into BOTH frames and compared with BOTH observed 2D
          segments -- neither of which it was given.

  POINTS  ORB match between the same two frames, triangulated from the same two
          poses by ray midpoint, reprojected into both frames the same way.

Same two views, same poses, each solved by its own pipeline. If lines are
mis-triangulated the reprojection fails in at least one frame, and the point
column says whether the geometry or the lines are at fault.
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
    th = np.clip(ru.copy(), 0, np.pi * 0.6)
    for _ in range(it):
        t2 = th * th
        f = th * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4) - ru
        df = 1 + 3*k1*t2 + 5*k2*t2**2 + 7*k3*t2**3 + 9*k4*t2**4
        th = np.clip(th - f / np.where(np.abs(df) < 1e-6, 1e-6, df), 0, np.pi * 0.6)
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    return b / np.linalg.norm(b, axis=-1, keepdims=True)


def proj(Pc, c):
    x, y, z = Pc[..., 0], Pc[..., 1], Pc[..., 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    k1, k2, k3, k4 = c["d"]
    td = th * (1 + k1*th**2 + k2*th**4 + k3*th**6 + k4*th**8)
    s = np.where(r > 1e-9, td / np.maximum(r, 1e-9), 0.0)
    return np.stack([c["fx"]*x*s + c["cx"], c["fy"]*y*s + c["cy"]], -1), th


def elsed(tool, d, f, out):
    Path("/tmp/_tf.list").write_text(f"{f}\n")
    subprocess.run([tool, str(d), "/tmp/_tf.list", out, "30", "15"],
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


def triangulate(n1_c, R1, t1, n2_c, R2, t2, min_deg=2.0):
    n1 = R1.T @ n1_c; n2 = R2.T @ n2_c
    C1 = -R1.T @ t1;  C2 = -R2.T @ t2
    d = np.cross(n1, n2); dn = np.linalg.norm(d)
    if dn < np.sin(np.deg2rad(min_deg)):
        return None, None
    d /= dn
    A = np.stack([n1, n2, d]); b = np.array([n1 @ C1, n2 @ C2, 0.0])
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    return d, np.cross(P, d)


def clip(d_w, m_w, R, t, b1, b2, maxd=200.0):
    d_c = R @ d_w; m_c = R @ m_w + np.cross(t, d_c)
    out = []
    for b in (b1, b2):
        cr = np.cross(b, d_c); den = cr @ cr
        if den < 1e-8:
            return None
        s = (m_c @ cr) / den
        if not np.isfinite(s) or s <= 0.05 or s > maxd:
            return None
        out.append(R.T @ (s * b - t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--elsed", required=True)
    ap.add_argument("--traj", required=True, help="f_orb.txt for the real poses")
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--gap", type=int, default=30, help="A->B, used to triangulate")
    ap.add_argument("--gap2", type=int, default=30, help="B->C, the HELD-OUT frame")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = Path(a.dataset); cam = load_cam(a.config, a.cam); T_bc = T_bc_of(a.config)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int); ft = np.atleast_1d(fr["t"])
    tof = dict(zip(fid, ft))
    d = np.loadtxt(a.traj); T = d[:, 0] / 1e9

    def pose(f):
        j = int(np.argmin(np.abs(T - tof[f])))
        if abs(T[j] - tof[f]) > 0.05:
            raise SystemExit(f"frame {f} (t={tof[f]:.3f}) has no pose in {a.traj}")
        Twb = np.eye(4)
        Twb[:3, :3] = Rotation.from_quat(d[j, 4:8]).as_matrix()
        Twb[:3, 3] = d[j, 1:4]
        Tcw = np.linalg.inv(Twb @ T_bc)
        return Tcw[:3, :3], Tcw[:3, 3]

    fA, fB, fC = a.frame, a.frame + a.gap, a.frame + a.gap + a.gap2
    RA, tA = pose(fA); RB, tB = pose(fB); RC, tC = pose(fC)
    CA, CB, CC = -RA.T @ tA, -RB.T @ tB, -RC.T @ tC
    print(f"triangulate from {fA} + {fB}  (baseline {np.linalg.norm(CB-CA):.3f} m)")
    print(f"JUDGE on held-out frame {fC}  (baseline from B {np.linalg.norm(CC-CB):.3f} m)\n")

    imA = cv2.imread(str(ds / f"cam{a.cam}" / f"{fA:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    imB = cv2.imread(str(ds / f"cam{a.cam}" / f"{fB:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    imC = cv2.imread(str(ds / f"cam{a.cam}" / f"{fC:06d}.jpg"), cv2.IMREAD_GRAYSCALE)

    # ---------------------------------------------------------------- LINES --
    sA, nA, dA, aA, b1A, b2A = lift(elsed(a.elsed, ds / f"cam{a.cam}", fA, "/tmp/_a.csv"), cam)
    sB, nB, dB, aB, b1B, b2B = lift(elsed(a.elsed, ds / f"cam{a.cam}", fB, "/tmp/_b.csv"), cam)
    sC, nC, dC, aC, b1C, b2C = lift(elsed(a.elsed, ds / f"cam{a.cam}", fC, "/tmp/_c.csv"), cam)
    asgBA = match(nB, dB, aB, nA, dA, aA)      # B -> A
    asgCB = match(nC, dC, aC, nB, dB, aB)      # C -> B
    triples = [(asgBA[i], i, k) for k, i in enumerate(asgCB)
               if i >= 0 and asgBA[i] >= 0]
    print(f"LINES  detected {len(sA)}/{len(sB)}/{len(sC)}, "
          f"tracked through all three: {len(triples)}")

    rows, rows_ab, segs3d = [], [], []
    for j, i, k in triples:
        dw, mw = triangulate(nA[j], RA, tA, nB[i], RB, tB)      # A + B only
        if dw is None:
            continue
        eA = clip(dw, mw, RA, tA, b1A[j], b2A[j])
        if eA is None:
            continue
        def resid(R, t, bb1, bb2):
            d_c = R @ dw; m_c = R @ mw + np.cross(t, d_c)
            npd = m_c / max(np.linalg.norm(m_c), 1e-12)
            return np.degrees(np.arcsin(np.clip(max(abs(npd @ bb1), abs(npd @ bb2)), 0, 1)))
        rows_ab.append(max(resid(RA, tA, b1A[j], b2A[j]),
                           resid(RB, tB, b1B[i], b2B[i])))   # fitted frames
        rows.append(resid(RC, tC, b1C[k], b2C[k]))           # HELD OUT
        segs3d.append((eA[0], eA[1]))
    rows = np.array(rows); rows_ab = np.array(rows_ab)

    # --------------------------------------------------------------- POINTS --
    orb = cv2.ORB_create(4000)
    kA, desA = orb.detectAndCompute(imA, None)
    kB, desB = orb.detectAndCompute(imB, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kC, desC = orb.detectAndCompute(imC, None)
    mAB = {m.queryIdx: m.trainIdx for m in bf.match(desA, desB)}
    mBC = {m.queryIdx: m.trainIdx for m in bf.match(desB, desC)}
    mt = [(qa, ib, mBC[ib]) for qa, ib in mAB.items() if ib in mBC]
    print(f"POINTS ORB tracked through all three: {len(mt)}")
    perr, perr_ab, pts3d = [], [], []
    for (qa, ib, ic) in mt:
        uA = np.array(kA[qa].pt); uB = np.array(kB[ib].pt); uC = np.array(kC[ic].pt)
        bA = unproj(uA[0], uA[1], cam); bB = unproj(uB[0], uB[1], cam)
        rA = RA.T @ bA; rB = RB.T @ bB                      # rays in world
        w = CB - CA
        aa, bb, cc = rA @ rA, rA @ rB, rB @ rB
        dd, ee = rA @ w, rB @ w
        den = aa*cc - bb*bb
        if abs(den) < 1e-9:
            continue
        s1 = (bb*ee - cc*dd) / -den
        s2 = (aa*ee - bb*dd) / -den
        if s1 <= 0 or s2 <= 0 or s1 > 200 or s2 > 200:
            continue
        X = 0.5 * ((CA + s1*rA) + (CB + s2*rB))       # from A + B only
        eab = 0.0
        for (R, t, u) in ((RA, tA, uA), (RB, tB, uB)):
            uv, th = proj(R @ X + t, cam)
            if th > np.deg2rad(95):
                eab = 1e9; break
            eab = max(eab, float(np.linalg.norm(uv - u)))
        uv, th = proj(RC @ X + tC, cam)                # HELD OUT
        if eab < 1e8 and th <= np.deg2rad(95):
            perr.append(float(np.linalg.norm(uv - uC)))
            perr_ab.append(eab); pts3d.append(X)
    perr = np.array(perr); perr_ab = np.array(perr_ab)
    print(f"POINTS triangulated {len(perr)}\n")

    px = np.degrees(np.arctan(1.0 / cam["fx"]))
    print(f"{'':44s}{'p25':>9s}{'p50':>9s}{'p75':>9s}{'p90':>9s}")
    print("  fitted frames A+B (zero expected for lines)")
    print(f"{'    LINE  residual [px-equiv]':44s}" +
          "".join(f"{np.percentile(rows_ab, q)/px:9.3f}" for q in (25, 50, 75, 90)))
    print(f"{'    POINT reprojection [px]':44s}" +
          "".join(f"{np.percentile(perr_ab, q):9.3f}" for q in (25, 50, 75, 90)))
    print(f"\n  HELD-OUT frame {fC} -- neither was fitted to it")
    print(f"{'    LINE  residual [px-equiv]':44s}" +
          "".join(f"{np.percentile(rows, q)/px:9.3f}" for q in (25, 50, 75, 90)))
    print(f"{'    POINT reprojection [px]':44s}" +
          "".join(f"{np.percentile(perr, q):9.3f}" for q in (25, 50, 75, 90)))
    print(f"\n(1 px = {px:.4f} deg)   lines evaluated {len(rows)}, points {len(perr)}")

    # ----------------------------------------------------------------- viz ---
    rr.init("two_frame_line_vs_point", spawn=False); rr.save(a.out)
    rr.set_time("t", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for nm, C, R in ((f"camA_f{fA}", CA, RA.T), (f"camB_f{fB}", CB, RB.T),
                     (f"camC_heldout_f{fC}", CC, RC.T)):
        rr.log(f"world/{nm}", rr.Transform3D(translation=C, mat3x3=R))
        rr.log(f"world/{nm}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]], image_plane_distance=0.4))
    if segs3d:
        rr.log("world/lines_triangulated", rr.LineStrips3D(
            [[p, q] for p, q in segs3d], colors=[[245, 210, 50]] * len(segs3d),
            radii=0.02))
    if len(pts3d):
        rr.log("world/points_triangulated", rr.Points3D(
            np.array(pts3d), colors=[[60, 230, 90]], radii=0.04))
    for nm, im, sg in ((f"imageA_f{fA}", imA, sA), (f"imageB_f{fB}", imB, sB),
                       (f"imageC_heldout_f{fC}", imC, sC)):
        v = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2) in sg:
            cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (60, 200, 255), 2)
        rr.log(nm, rr.Image(cv2.rotate(v, cv2.ROTATE_180)).compress(jpeg_quality=85))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
