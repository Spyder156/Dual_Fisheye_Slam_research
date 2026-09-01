#!/usr/bin/env python
"""One line, three frames, end-to-end consistency -- verifiable in 2D.

The claim being tested: if the wiring is correct, then

  * the line's two ENDPOINTS put through the POINT math, and
  * the line itself put through the LINE math

must land in the same place in 3D, and BOTH must reproject onto the detected
segment in all three frames.

The 2D reprojection is the part that cannot be argued with. A 3D view can look
plausible from a lucky angle; a curve drawn on top of the image either sits on
the detected line or it does not.

Note a fisheye projects a straight world line to a CURVE, so the reprojected
line is drawn by sampling points along the 3D segment and projecting each one.
Projecting only the two endpoints and joining them with a straight image line is
wrong on this camera and was one reason earlier pictures looked inconsistent.
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
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return b[0] if b.ndim == 2 and b.shape[0] == 1 and np.isscalar(u) else b


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
    Path("/tmp/_cl.list").write_text(f"{f}\n")
    subprocess.run([tool, str(dd), "/tmp/_cl.list", out, "30", "15"],
                   capture_output=True, check=True)
    return np.array([[float(x) for x in l.split(",")[1:5]]
                     for l in open(out).read().splitlines()[1:]])


def lift(segs, cam, mn=np.deg2rad(1.5), mx=np.deg2rad(40.0), max_theta=np.deg2rad(70.0)):
    """Lift to bearings, and REFUSE the rim.

    Every segment in the first pass sat at theta = 88-93 deg from the optical
    axis -- the extreme edge of the fisheye, one of them past 90 deg, i.e.
    outside the half-field the model is calibrated for. Sorting by angular
    length selects for exactly those, because on a fisheye the longest arcs are
    the rim ones. A bearing there is barely constrained, so the interpretation
    plane built from two of them is barely constrained either.
    """
    b1 = unproj(segs[:, 0], segs[:, 1], cam)
    b2 = unproj(segs[:, 2], segs[:, 3], cam)
    n = np.cross(b1, b2)
    ln = np.linalg.norm(n, axis=1)
    ang = np.arcsin(np.clip(ln, 0, 1))
    th1 = np.arccos(np.clip(b1[:, 2], -1, 1))
    th2 = np.arccos(np.clip(b2[:, 2], -1, 1))
    k = (ln > 1e-9) & (ang >= mn) & (ang <= mx) & (th1 < max_theta) & (th2 < max_theta)
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


def line_from_planes(normals, poses):
    """Least-squares line through N interpretation planes -- no 2-view gate.

    Each observation says: the line lies in the plane through camera centre C
    with normal n. Direction d must be orthogonal to every n (smallest singular
    vector); a point P on the line satisfies n.P = n.C for every view.
    """
    N = np.stack([R.T @ n for n, (R, t) in zip(normals, poses)])
    C = np.stack([-R.T @ t for (R, t) in poses])
    _, _, Vt = np.linalg.svd(N)
    d = Vt[-1] / np.linalg.norm(Vt[-1])
    A = np.vstack([N, d[None, :]])
    b = np.concatenate([np.einsum("ij,ij->i", N, C), [0.0]])
    P, *_ = np.linalg.lstsq(A, b, rcond=None)
    return d, np.cross(P, d)


def clip(d_w, m_w, R, t, b1, b2, maxd=1000.0, mirror=True):
    """Intersect the two endpoint bearings with the line.

    A negative depth means the reconstruction ended up on the far side of the
    camera. The line is visible in the image, so the bearing is right and only
    the side is wrong; taking |s| puts it back at the same distance in front.
    That is a patch on a bad reconstruction, not a fix -- kept so the result can
    be SEEN instead of silently dropped.
    """
    d_c = R @ d_w; m_c = R @ m_w + np.cross(t, d_c)
    out, ss = [], []
    for b in (b1, b2):
        cr = np.cross(b, d_c); den = cr @ cr
        if den < 1e-12:
            return None, None
        s = (m_c @ cr) / den
        if not np.isfinite(s):
            return None, None
        ss.append(s)
        if mirror:
            s = abs(s)
        if s <= 0.02 or s > maxd:
            return None, None
        out.append(R.T @ (s * b - t))
    return out, ss


def tri_point(bs, poses):
    """Least-squares point from N bearing rays."""
    A = np.zeros((3, 3)); c = np.zeros(3)
    for b, (R, t) in zip(bs, poses):
        r = R.T @ b; C = -R.T @ t
        M = np.eye(3) - np.outer(r, r)
        A += M; c += M @ C
    try:
        return np.linalg.solve(A, c)
    except np.linalg.LinAlgError:
        return None


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
    ap.add_argument("--max-theta", type=float, default=70.0,
                    help="reject bearings beyond this angle from the optical axis")
    ap.add_argument("--pick", type=int, default=-1,
                    help="-1 = the line with the best 3-frame reprojection")
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
    IM = [cv2.imread(str(ds / f"cam{a.cam}" / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE) for f in F]
    L = [lift(elsed(a.elsed, ds / f"cam{a.cam}", f, f"/tmp/_c{i}.csv"), cam,
              max_theta=np.deg2rad(a.max_theta))
         for i, f in enumerate(F)]
    CC = [-R.T @ t for (R, t) in POSE]
    print(f"frames {F}   baselines A->B {np.linalg.norm(CC[1]-CC[0]):.2f} m, "
          f"B->C {np.linalg.norm(CC[2]-CC[1]):.2f} m")

    a10 = match(L[1][1], L[1][2], L[1][3], L[0][1], L[0][2], L[0][3])
    a21 = match(L[2][1], L[2][2], L[2][3], L[1][1], L[1][2], L[1][3])
    triples = [(a10[i1], i1, i2) for i2, i1 in enumerate(a21)
               if i1 >= 0 and a10[i1] >= 0]
    if not triples:
        raise SystemExit("no line persists through all three frames")

    FAIL = {"clip": 0, "tri_point": 0, "behind": 0}

    def evaluate(tr):
        """Solve from ALL THREE views, then measure the 2D reprojection error."""
        normals = [L[k][1][tr[k]] for k in range(3)]
        dw, mw = line_from_planes(normals, POSE)
        seg, ss = clip(dw, mw, POSE[0][0], POSE[0][1],
                       L[0][4][tr[0]], L[0][5][tr[0]], mirror=True)
        if seg is None:
            FAIL["clip"] += 1
            return None
        dn = max(abs(dw @ (R.T @ n)) for n, (R, t) in zip(normals, POSE))
        # endpoints through the POINT math, from all three views
        E = []
        for e in (0, 1):
            bs = [L[k][4 + e][tr[k]] for k in range(3)]
            X = tri_point(bs, POSE)
            if X is None:
                FAIL["tri_point"] += 1
                return None
            E.append(X)
        # 2D error: sample the 3D segment, project, distance to the observed 2D line
        err = []
        for k in range(3):
            R, t = POSE[k]
            samp = np.stack([seg[0] + s * (seg[1] - seg[0]) for s in np.linspace(0, 1, 15)])
            uv, th = proj((R @ samp.T).T + t, cam)
            if (th > np.deg2rad(95)).all():
                FAIL["behind"] += 1
                return None
            uv = uv[th <= np.deg2rad(95)]
            x1, y1, x2, y2 = L[k][0][tr[k]]
            u = np.array([x2 - x1, y2 - y1], float); u /= max(np.linalg.norm(u), 1e-9)
            nrm = np.array([-u[1], u[0]])
            err.append(float(np.abs((uv - np.array([x1, y1])) @ nrm).max()))
        return dw, mw, seg, E, err, dn, ss

    scored = []
    for q, tr in enumerate(triples):
        r = evaluate(tr)
        if r is not None:
            scored.append((max(r[4]), q, tr, r))
    if not scored:
        raise SystemExit(f"no line survived out of {len(triples)}; reasons: {FAIL}")
    # rank by GEOMETRIC CONSISTENCY first: d.n is how far the three planes are
    # from actually sharing a line. A small reprojection error on an
    # inconsistent triple is luck.
    scored.sort(key=lambda z: (z[3][5], z[0]))

    print(f"\n{len(scored)}/{len(triples)} lines solved. "
          f"max perpendicular reprojection error over the 3 frames [px]:")
    print(f"{'rank':>5s}{'angLen':>9s}{'d.n':>8s}{'depthA':>9s}"
          f"{'f'+str(F[0]):>9s}{'f'+str(F[1]):>9s}{'f'+str(F[2]):>9s}{'mirrored':>10s}")
    for rank, (mx, q, tr, r) in enumerate(scored[:12]):
        mir = "yes" if min(r[6]) < 0 else "no"
        print(f"{rank:5d}{np.degrees(L[0][3][tr[0]]):8.2f}d{r[5]:8.3f}"
              f"{abs(r[6][0]):9.2f}" + "".join(f"{e:9.2f}" for e in r[4]) + f"{mir:>10s}")

    sel = 0 if a.pick < 0 else min(a.pick, len(scored) - 1)
    mx, q, tr, (dw, mw, seg, E, err, dn, ss) = scored[sel]
    print(f"\ngeometric consistency d.n = {dn:.4f} (0 = the three planes share a line)")
    print(f"raw intersection depths from camera A: {ss[0]:.3f}, {ss[1]:.3f} m"
          + ("   <- NEGATIVE, mirrored through the camera" if min(ss) < 0 else ""))
    print(f"\nusing rank {sel}: reprojection error {['%.2f' % e for e in err]} px")
    print(f"3D endpoints from the LINE math : {np.round(seg[0],3)}  {np.round(seg[1],3)}")
    print(f"3D endpoints from the POINT math: {np.round(E[0],3)}  {np.round(E[1],3)}")
    print(f"gap endpoint1 {np.linalg.norm(E[0]-seg[0]):.4f} m,  "
          f"endpoint2 {np.linalg.norm(E[1]-seg[1]):.4f} m")
    print(f"depth from camera A: {np.linalg.norm(0.5*(seg[0]+seg[1]) - CC[0]):.2f} m, "
          f"3D length {np.linalg.norm(seg[1]-seg[0]):.3f} m")

    # ------------------------------------------------------------------ viz --
    S = float(max(np.linalg.norm(0.5*(seg[0]+seg[1]) - c) for c in CC))
    rr.init("consistent_line", spawn=False); rr.save(a.out)
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
        rr.log(f"world/ray_f{F[k]}", rr.LineStrips3D(
            [[CC[k], E[0]], [CC[k], E[1]]], colors=[COL[k]] * 2, radii=0.0012 * S))
    rr.log("world/LINE_math", rr.LineStrips3D(
        [[seg[0], seg[1]]], colors=[[245, 210, 50]], radii=0.008 * S))
    rr.log("world/POINT_math", rr.Points3D(
        np.stack(E), colors=[[60, 230, 90]] * 2, radii=0.014 * S))

    # 2D proof: the reprojected line and endpoints drawn over each image
    for k in range(3):
        R, t = POSE[k]
        v = cv2.cvtColor(IM[k], cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = L[k][0][tr[k]]
        cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 7, cv2.LINE_AA)
        samp = np.stack([seg[0] + s * (seg[1] - seg[0]) for s in np.linspace(-0.3, 1.3, 60)])
        uv, th = proj((R @ samp.T).T + t, cam)
        good = th < np.deg2rad(95)
        pts = uv[good].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(v, [pts.reshape(-1, 1, 2)], False, (50, 210, 245), 3, cv2.LINE_AA)
        for X in E:
            p, thp = proj((R @ X + t)[None, :], cam)
            if thp[0] < np.deg2rad(95):
                cv2.circle(v, (int(p[0, 0]), int(p[0, 1])), 11, (80, 240, 90), -1)
        rr.log(f"frame{k}_f{F[k]}",
               rr.Image(cv2.rotate(v, cv2.ROTATE_180)).compress(jpeg_quality=92))

    print("\nWHITE thick = detected 2D segment | YELLOW curve = reprojected 3D LINE")
    print("GREEN dots  = reprojected 3D endpoints (POINT math)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
