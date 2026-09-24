#!/usr/bin/env python
"""Project the VERIFIED mapped 3D lines back onto the actual video frames.

The structural test: a doorway's edges must be present, correctly placed and
STABLE across frames -- same id, same edge, no sliding. This uses only the
estimator's outputs (ml_orb.csv = verified lines, f_orb.txt = frame poses) and
the calibrated forward model; nothing is re-detected.

Conventions:
  f_orb.txt        t[ns], p, q(xyzw): T_w_b (body pose in world)
  IMU.T_b_c1       4x4, cam0 -> body   (p_b = T_b_c1 @ p_c)
  Rig.T_c0_c1      4x4, cam1 -> cam0
  ml_orb.csv       endpoints in the b0 frame == f_orb world frame

Usage: doorway_overlay.py <rundir> --dataset D --config C --out DIR
                          [--times t1,t2,... | auto] [--cam 0]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from debug_line_projection import load_cam  # noqa: E402


def project_kb4(P_c, cam):
    """KB4 forward model. P_c: Nx3 camera-frame points -> Nx2 pixels."""
    x, y, z = P_c[:, 0], P_c[:, 1], P_c[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    k1, k2, k3, k4 = cam["d"]
    rd = th * (1 + k1*th**2 + k2*th**4 + k3*th**6 + k4*th**8)
    s = np.where(r > 1e-9, rd / np.maximum(r, 1e-9), 0.0)
    return np.stack([cam["fx"] * x * s + cam["cx"],
                     cam["fy"] * y * s + cam["cy"]], -1)


def load_mat(cfg_txt, key):
    import re
    m = re.search(rf"{key}:.*?data:\s*\[(.*?)\]", cfg_txt, re.S)
    if not m:
        return None
    v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v).reshape(4, 4)


def quat_R(q):  # xyzw -> R
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--times", default="auto",
                    help="comma-separated timestamps [s], or 'auto' = the 3 "
                         "spaced frames seeing the most verified lines")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg_txt = args.config.read_text()
    cam = load_cam(args.config, args.cam)
    T_b_c0 = load_mat(cfg_txt, "IMU.T_b_c1")
    T_c0_c1 = load_mat(cfg_txt, "Rig.T_c0_c1")

    # frame poses: T_w_b per timestamp
    tr = np.loadtxt(args.rundir / "f_orb.txt")
    tf = tr[:, 0] / 1e9
    # verified lines
    ml = np.atleast_1d(np.genfromtxt(args.rundir / "ml_orb.csv", delimiter=",", names=True))
    segs = [(int(r["id"]), np.array([r["x1"], r["y1"], r["z1"]]),
             np.array([r["x2"], r["y2"], r["z2"]])) for r in ml]
    # frame index: filename by row number in frames.csv
    fr = np.genfromtxt(args.dataset / "frames.csv", delimiter=",", names=True)

    def T_c_w(row):
        R_wb, p_wb = quat_R(tr[row, 4:8]), tr[row, 1:4]
        T_w_b = np.eye(4); T_w_b[:3, :3] = R_wb; T_w_b[:3, 3] = p_wb
        T = np.linalg.inv(T_b_c0) @ np.linalg.inv(T_w_b)      # world -> cam0
        if args.cam == 1:
            T = np.linalg.inv(T_c0_c1) @ T                    # world -> cam1
        return T

    def visible_lines(row):
        T = T_c_w(row)
        vis = []
        for lid, e1, e2 in segs:
            S = np.linspace(0, 1, 32)
            P = (1 - S)[:, None] * e1 + S[:, None] * e2
            Pc = (T[:3, :3] @ P.T).T + T[:3, 3]
            ang = np.arctan2(np.hypot(Pc[:, 0], Pc[:, 1]), Pc[:, 2])
            ok = ang < np.radians(95)                          # inside the lens FOV
            if ok.sum() < 8:
                continue
            uv = project_kb4(Pc[ok], cam)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"]) & \
                     (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"])
            if inside.sum() >= 8:
                vis.append((lid, uv[inside]))
        return vis

    if args.times == "auto":
        rows = range(0, len(tf), 10)
        scored = [(len(visible_lines(r)), r) for r in rows]
        scored.sort(reverse=True)
        picks, chosen = [], []
        for cnt, r in scored:
            if all(abs(tf[r] - tf[c]) > 1.5 for c in chosen):
                chosen.append(r); picks.append((cnt, r))
            if len(chosen) == 3:
                break
    else:
        picks = []
        for ts in [float(x) for x in args.times.split(",")]:
            r = int(np.argmin(np.abs(tf - ts)))
            picks.append((None, r))

    rng = np.random.default_rng(0)
    colors = {lid: tuple(int(c) for c in rng.integers(80, 255, 3))
              for lid, _, _ in segs}
    for cnt, r in picks:
        # frame image whose timestamp matches this pose
        fi = int(np.argmin(np.abs(fr["t"] - tf[r])))
        img_p = args.dataset / f"cam{args.cam}" / f"{int(fr['frame'][fi]):06d}.jpg"
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"missing frame {img_p}"); continue
        for lid, uv in visible_lines(r):
            pts = uv.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, colors[lid], 2, cv2.LINE_AA)
            cv2.putText(img, str(lid), tuple(pts[len(pts)//2, 0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[lid], 2, cv2.LINE_AA)
        outp = args.out / f"overlay_cam{args.cam}_t{tf[r]:.2f}.png"
        cv2.imwrite(str(outp), img)
        print(f"{outp}  ({len(visible_lines(r))} verified lines drawn)")


if __name__ == "__main__":
    main()
