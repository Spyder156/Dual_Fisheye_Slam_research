#!/usr/bin/env python
"""Dual-fisheye front end: KLT point tracks + spherical ELSED line tracks.

Built to the winning entry's recipe (ACDC-VSLAM, #1, 2410.0):

    "Spherical line features. ELSED segments, endpoints lifted to bearings,
     represented by the normal of the associated great circle. Filtered by
     angular length; tracked with gating on direction, normal alignment,
     length consistency."

WHY BOTH, NOT LINES INSTEAD OF POINTS: every team that uses lines uses them
ALONGSIDE points. Points give well-localised 2-DoF constraints where texture
exists; lines give 1-DoF constraints that survive where texture does not.
Measured on our own failure (floor_EG run_2, t=10156.8..10157.8):

    tracked POINTS  : 0-2 per frame   (ORB-SLAM3 collapses)
    detected LINES  : 643-993 per frame per camera

THE GEOMETRY. A straight world line projects on a central camera to a GREAT
CIRCLE on the unit sphere, described by its plane normal

    n = b1 x b2 / |b1 x b2|

and every bearing b on the line satisfies n . b = 0. That residual is invariant
to sliding ALONG the line, which is the aperture problem stated honestly: a line
carries one constraint, not two. Tracking therefore never tries to localise a
point along an edge -- it matches the plane.

ANGULAR length, not pixel length. A fisheye compresses the rim, so a segment
that is short in pixels can span a large angle. Filtering in pixels discards
exactly the wide-baseline lines that constrain rotation best.
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------- camera ----


def load_cam(path, cam=0):
    blk = re.split(r"\ncam\d+:", Path(path).read_text())[1 + cam]
    fx, fy, cx, cy = [float(x) for x in
                      re.search(r"intrinsics:\s*\[(.*?)\]", blk).group(1).split(",")]
    d = [float(x) for x in
         re.search(r"distortion_coeffs:\s*\[(.*?)\]", blk).group(1).split(",")]
    w, h = [int(float(x)) for x in
            re.search(r"resolution:\s*\[(.*?)\]", blk).group(1).split(",")]
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, d=d, w=w, h=h)


def unproject_kb4(u, v, cam, iters=10):
    """pixel -> unit bearing (KB4), plus a validity mask.

    The Newton solve diverges for pixels outside the lens's real field of view
    (image CORNERS on a fisheye are outside the image circle). Those must be
    reported invalid, not silently clipped -- an unchecked unprojection is
    exactly the bug that gave OKVIS2 false cross-camera overlap.
    """
    mx = (np.asarray(u, float) - cam["cx"]) / cam["fx"]
    my = (np.asarray(v, float) - cam["cy"]) / cam["fy"]
    ru = np.hypot(mx, my)
    k1, k2, k3, k4 = cam["d"]
    th = np.clip(ru.copy(), 0, np.pi * 0.6)
    for _ in range(iters):
        th2 = th * th
        f = th * (1 + k1*th2 + k2*th2**2 + k3*th2**3 + k4*th2**4) - ru
        df = 1 + 3*k1*th2 + 5*k2*th2**2 + 7*k3*th2**3 + 9*k4*th2**4
        df = np.where(np.abs(df) < 1e-6, np.sign(df) * 1e-6 + 1e-6, df)
        th = np.clip(th - f / df, 0, np.pi * 0.6)
    valid = np.isfinite(th) & (th < np.deg2rad(95.0)) & np.isfinite(ru)
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    valid &= (n[..., 0] > 1e-9)
    return b / np.maximum(n, 1e-12), valid


# ------------------------------------------------------------------ lines ----


class SphericalLines:
    """ELSED segments -> great-circle normals, tracked with three gates."""

    def __init__(self, cam, min_ang_deg=1.5,
                 gate_normal_deg=3.0, gate_dir_deg=8.0, gate_len_ratio=0.5):
        self.cam = cam
        self.min_ang = np.deg2rad(min_ang_deg)
        # the winner's three gates
        self.g_normal = np.deg2rad(gate_normal_deg)   # plane-normal alignment
        self.g_dir = np.deg2rad(gate_dir_deg)         # in-plane direction
        self.g_len = gate_len_ratio                   # angular-length consistency
        self.tracks = {}      # id -> dict(n, dir, ang, last_seen, len)
        self._next = 0

    def lift(self, segs):
        """(N,4) pixel segments -> (normals, dirs, angular lengths)."""
        if segs is None or len(segs) == 0:
            return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        s = np.asarray(segs, float).reshape(-1, 4)
        b1, v1 = unproject_kb4(s[:, 0], s[:, 1], self.cam)
        b2, v2 = unproject_kb4(s[:, 2], s[:, 3], self.cam)
        n = np.cross(b1, b2)
        ln = np.linalg.norm(n, axis=1)
        ang = np.arcsin(np.clip(ln, 0, 1))
        ok = v1 & v2 & (ln > 1e-9) & (ang >= self.min_ang)
        n = n[ok] / ln[ok, None]
        # in-plane direction: endpoint-to-endpoint on the sphere
        d = b2[ok] - b1[ok]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return n, d, ang[ok]

    def track(self, segs, frame_id):
        """Match this frame's lines to live tracks. Returns (n_matched, n_new)."""
        n, d, ang = self.lift(segs)
        if len(n) == 0:
            return 0, 0
        ids = list(self.tracks.keys())
        matched = 0
        used = np.zeros(len(n), bool)
        if ids:
            N = np.stack([self.tracks[i]["n"] for i in ids])
            D = np.stack([self.tracks[i]["dir"] for i in ids])
            A = np.array([self.tracks[i]["ang"] for i in ids])
            # gate 1: normal alignment (sign-free -- a line has no orientation)
            cn = np.abs(N @ n.T)
            # gate 2: in-plane direction
            cd = np.abs(D @ d.T)
            # gate 3: angular-length consistency
            lr = np.minimum(A[:, None], ang[None, :]) / np.maximum(
                np.maximum(A[:, None], ang[None, :]), 1e-12)
            ok = ((cn >= np.cos(self.g_normal)) &
                  (cd >= np.cos(self.g_dir)) &
                  (lr >= self.g_len))
            score = np.where(ok, cn, -1.0)
            for ti in np.argsort(-score.max(axis=1)):
                j = int(np.argmax(score[ti]))
                if score[ti, j] < 0 or used[j]:
                    continue
                used[j] = True
                t = self.tracks[ids[ti]]
                t.update(n=n[j], dir=d[j], ang=ang[j], last_seen=frame_id)
                t["len"] += 1
                matched += 1
        for j in np.where(~used)[0]:
            self.tracks[self._next] = dict(n=n[j], dir=d[j], ang=ang[j],
                                           last_seen=frame_id, len=1)
            self._next += 1
        # retire stale tracks
        for k in [k for k, v in self.tracks.items() if frame_id - v["last_seen"] > 3]:
            del self.tracks[k]
        return matched, int((~used).sum())


# ------------------------------------------------------------------ points ---


class KLTPoints:
    """Plain fast KLT. Points where texture exists; lines carry the rest."""

    def __init__(self, max_pts=1200, quality=0.01, min_dist=12, mask=None):
        self.max_pts, self.quality, self.min_dist = max_pts, quality, min_dist
        self.mask = mask
        self.prev = None
        self.pts = None

    def step(self, img):
        if self.prev is None or self.pts is None or len(self.pts) < 50:
            m = None
            if self.mask is not None:
                m = (self.mask == 0).astype(np.uint8) * 255   # nonzero = ignore
            self.pts = cv2.goodFeaturesToTrack(img, self.max_pts, self.quality,
                                               self.min_dist, mask=m)
            self.prev = img
            return 0, (0 if self.pts is None else len(self.pts))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, img, self.pts, None,
                                              winSize=(21, 21), maxLevel=4)
        st = st.ravel().astype(bool)
        tracked = int(st.sum())
        self.pts = nxt[st].reshape(-1, 1, 2) if tracked else None
        self.prev = img
        return tracked, 0


# -------------------------------------------------------------------- main ---


def run_elsed(tool, frames_dir, ids, out_csv, grad=30, minlen=15):
    lst = Path(out_csv).with_suffix(".list")
    lst.write_text("\n".join(str(i) for i in ids) + "\n")
    r = subprocess.run([tool, str(frames_dir), str(lst), str(out_csv),
                        str(grad), str(minlen)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"line_extract failed: {r.stderr[-400:]}")
    print("  " + r.stderr.strip().splitlines()[-1])
    d = {}
    with open(out_csv) as f:
        f.readline()
        for l in f:
            p = l.split(",")
            d.setdefault(int(p[0]), []).append([float(x) for x in p[1:5]])
    return {k: np.array(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--imucam", required=True)
    ap.add_argument("--elsed", required=True, help="line_extract binary")
    ap.add_argument("--frames", default=None, help="first:last")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--min-ang", type=float, default=1.5)
    a = ap.parse_args()

    ds = Path(a.dataset)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int)
    ft = np.atleast_1d(fr["t"])
    if a.frames:
        lo, hi = (int(x) for x in a.frames.split(":"))
        m = (fid >= lo) & (fid <= hi); fid, ft = fid[m], ft[m]

    rows = []
    for c in (0, 1):
        cam = load_cam(a.imucam, c)
        print(f"cam{c}: ELSED")
        segs = run_elsed(a.elsed, ds / f"cam{c}", fid, out / f"segs_cam{c}.csv")
        sl = SphericalLines(cam, min_ang_deg=a.min_ang)
        kp = KLTPoints()
        for k, t in zip(fid, ft):
            img = cv2.imread(str(ds / f"cam{c}" / f"{k:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ptr, pnew = kp.step(img)
            lm, ln = sl.track(segs.get(int(k), np.zeros((0, 4))), int(k))
            rows.append((t, c, ptr, lm, len(sl.tracks)))
    r = np.array(rows)
    np.savetxt(out / "frontend_health.csv", r, delimiter=",",
               header="t,cam,klt_tracked,lines_matched,line_tracks", comments="")
    lab = a.label or ds.name
    for c in (0, 1):
        m = r[:, 1] == c
        if not m.any():
            continue
        print(f"{lab:16s} cam{c}  KLT tracked med {np.median(r[m,2]):6.0f} "
              f"min {r[m,2].min():5.0f} | LINES matched med {np.median(r[m,3]):6.0f} "
              f"min {r[m,3].min():5.0f} | live line tracks med {np.median(r[m,4]):6.0f}")
    print(f"-> {out/'frontend_health.csv'}")


if __name__ == "__main__":
    main()
