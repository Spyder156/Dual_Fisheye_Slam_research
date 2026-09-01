#!/usr/bin/env python
"""One frame, one rerun: do POINTS and LINES land in the same 3D place?

No SLAM, no trajectory, no score. Just the projective math, isolated.

For each selected 2D segment we build its 3D position TWICE, by the two paths
the pipeline actually uses, and draw both:

  POINT PATH (green)   endpoint pixel -> KB4 unprojection -> unit bearing
                       -> place at a chosen depth -> camera frame -> world.
                       This is exactly what a MapPoint does.

  LINE PATH (yellow)   the same two bearings -> great-circle normal n=b1xb2
                       -> Plucker line (d, m) -> transform world<->camera as
                       m_c = R*m + t x (R*d) -> intersect the endpoint bearings
                       back onto the line via
                           s = (m_c . (b x d_c)) / |b x d_c|^2
                       This is MapLine + SetExtentFromBearings.

If the two agree, the line math is consistent with the point math and any bug
lies further upstream. If they disagree, the difference is visible directly.

A deliberately NON-TRIVIAL camera pose is used (rotation + translation), because
with an identity pose a transposed rotation or a sign-flipped translation looks
correct.

TRIANGULATION CHECK (magenta): the same line is projected into a second camera
with a real baseline, and MapLine::Triangulate is re-run from the two normals.
The recovered line should coincide with the truth.
"""
import argparse
import re
import subprocess
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


def unproject_kb4(u, v, cam, iters=15):
    mx = (np.asarray(u, float) - cam["cx"]) / cam["fx"]
    my = (np.asarray(v, float) - cam["cy"]) / cam["fy"]
    ru = np.hypot(mx, my)
    k1, k2, k3, k4 = cam["d"]
    th = np.clip(ru.copy(), 0, np.pi * 0.6)
    for _ in range(iters):
        t2 = th * th
        f = th * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4) - ru
        df = 1 + 3*k1*t2 + 5*k2*t2**2 + 7*k3*t2**3 + 9*k4*t2**4
        th = np.clip(th - f / np.where(np.abs(df) < 1e-6, 1e-6, df), 0, np.pi * 0.6)
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    return b / np.linalg.norm(b, axis=-1, keepdims=True)


def extent_from_bearings(d_w, m_w, Rcw, tcw, b1, b2):
    """MapLine::SetExtentFromBearings, verbatim."""
    d_c = Rcw @ d_w
    m_c = Rcw @ m_w + np.cross(tcw, d_c)
    out = []
    for b in (b1, b2):
        cr = np.cross(b, d_c)
        den = cr @ cr
        if den < 1e-8:
            return None
        s = (m_c @ cr) / den
        if not np.isfinite(s) or s <= 0.05 or s > 500:
            return None
        out.append(Rcw.T @ (s * b - tcw))
    return out


def triangulate(n1_c, Rcw1, tcw1, n2_c, Rcw2, tcw2, min_deg=2.0):
    """MapLine::Triangulate, verbatim."""
    n1_w = Rcw1.T @ n1_c
    n2_w = Rcw2.T @ n2_c
    C1 = -Rcw1.T @ tcw1
    C2 = -Rcw2.T @ tcw2
    d = np.cross(n1_w, n2_w)
    dn = np.linalg.norm(d)
    if dn < np.sin(np.deg2rad(min_deg)):
        return None, None
    d = d / dn
    A = np.stack([n1_w, n2_w, d])
    b = np.array([n1_w @ C1, n2_w @ C2, 0.0])
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    return d, np.cross(P, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--elsed", required=True)
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--n-lines", type=int, default=14)
    ap.add_argument("--depth", type=float, default=4.0, help="depth for endpoint 1")
    ap.add_argument("--depth2", type=float, default=6.0, help="depth for endpoint 2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = Path(a.dataset)
    cam = load_cam(a.config, a.cam)
    img = cv2.imread(str(ds / f"cam{a.cam}" / f"{a.frame:06d}.jpg"), cv2.IMREAD_GRAYSCALE)

    lst = Path("/tmp/_dbg.list"); lst.write_text(f"{a.frame}\n")
    subprocess.run([a.elsed, str(ds / f"cam{a.cam}"), str(lst), "/tmp/_dbg.csv", "30", "15"],
                   capture_output=True, check=True)
    segs = np.array([[float(x) for x in l.split(",")[1:5]]
                     for l in open("/tmp/_dbg.csv").read().splitlines()[1:]])
    # longest segments, spread over the image, so the picture is readable
    order = np.argsort(-np.hypot(segs[:, 2]-segs[:, 0], segs[:, 3]-segs[:, 1]))
    segs = segs[order[:a.n_lines]]
    print(f"frame {a.frame} cam{a.cam}: drawing {len(segs)} of "
          f"{len(order)} detected segments")

    # A deliberately non-trivial pose. With identity, a transposed rotation or a
    # sign-flipped translation still looks right.
    Rwc = Rotation.from_euler("zyx", [25.0, -15.0, 8.0], degrees=True).as_matrix()
    Cw = np.array([1.5, -2.0, 0.7])           # camera centre in world
    Rcw = Rwc.T
    tcw = -Rcw @ Cw                            # world -> camera

    # second camera, real baseline, for the triangulation check
    Rwc2 = Rotation.from_euler("zyx", [22.0, -12.0, 8.0], degrees=True).as_matrix()
    Cw2 = Cw + np.array([0.45, 0.30, 0.05])
    Rcw2 = Rwc2.T
    tcw2 = -Rcw2 @ Cw2

    rr.init("line_vs_point_projection", spawn=False)
    rr.save(a.out)
    rr.set_time("t", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for nm, C, R in (("cam0", Cw, Rwc), ("cam1_triangulation", Cw2, Rwc2)):
        rr.log(f"world/{nm}", rr.Transform3D(translation=C, mat3x3=R))
        rr.log(f"world/{nm}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]], image_plane_distance=0.5))

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    nbad = 0
    for i, (x1, y1, x2, y2) in enumerate(segs):
        b1 = unproject_kb4(x1, y1, cam)
        b2 = unproject_kb4(x2, y2, cam)

        # ---- POINT PATH: bearing * depth, camera -> world -------------------
        P1 = Rcw.T @ (a.depth * b1 - tcw)
        P2 = Rcw.T @ (a.depth2 * b2 - tcw)

        # ---- LINE PATH: those two points define the Plucker line ------------
        d_w = P2 - P1
        d_w = d_w / np.linalg.norm(d_w)
        m_w = np.cross(P1, d_w)                 # m = p x d
        ext = extent_from_bearings(d_w, m_w, Rcw, tcw, b1, b2)

        col = tuple(int(v) for v in np.random.default_rng(i).integers(80, 255, 3))
        rr.log(f"world/rays/{i}", rr.LineStrips3D(
            [[Cw, P1], [Cw, P2]], colors=[[70, 70, 80]] * 2, radii=0.004))
        rr.log(f"world/point_path/{i}", rr.Points3D(
            np.stack([P1, P2]), colors=[[60, 230, 90]] * 2, radii=0.05))
        if ext is None:
            nbad += 1
        else:
            rr.log(f"world/line_path/{i}", rr.LineStrips3D(
                [[ext[0], ext[1]]], colors=[[245, 210, 50]], radii=0.025))
            rr.log(f"world/line_path_ends/{i}", rr.Points3D(
                np.stack(ext), colors=[[245, 160, 40]] * 2, radii=0.035))

        # ---- TRIANGULATION CHECK from two views -----------------------------
        n1_c = np.cross(b1, b2); n1_c /= np.linalg.norm(n1_c)
        q1 = Rcw2 @ P1 + tcw2
        q2 = Rcw2 @ P2 + tcw2
        n2_c = np.cross(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))
        n2_c /= np.linalg.norm(n2_c)
        dt, mt = triangulate(n1_c, Rcw, tcw, n2_c, Rcw2, tcw2)
        if dt is not None:
            et = extent_from_bearings(dt, mt, Rcw, tcw, b1, b2)
            if et is not None:
                rr.log(f"world/triangulated/{i}", rr.LineStrips3D(
                    [[et[0], et[1]]], colors=[[230, 80, 230]], radii=0.015))

        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3, cv2.LINE_AA)

    rr.log("image", rr.Image(cv2.rotate(vis, cv2.ROTATE_180)).compress(jpeg_quality=90))
    print(f"line path failed on {nbad}/{len(segs)} segments")
    print(f"\nGREEN dots   = endpoints via the POINT math (bearing x depth)")
    print(f"YELLOW line  = the same line via the LINE math (Plucker + bearing intersect)")
    print(f"MAGENTA line = re-triangulated from two views")
    print(f"grey rays    = camera centre -> endpoint\n-> {a.out}")


if __name__ == "__main__":
    main()
