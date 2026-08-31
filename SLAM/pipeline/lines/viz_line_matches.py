#!/usr/bin/env python
"""Side-by-side line MATCHING visualisation -- the pipeline's own flow, drawn.

Mirrors LineExtractor exactly, step for step, so what you see is what the
estimator matched and not a second implementation that happens to agree:

  1. ELSED segments                     (the same line_extract binary)
  2. self-occlusion mask                (drop a segment if EITHER endpoint is on
                                         the rig's own hardware)
  3. endpoints -> unit bearings (KB4)   -> great-circle normal n = b1 x b2
  4. angular-length filter              [minAngLen, maxAngLen]
  5. three gates, sign-free             normal 3 deg, direction 8 deg,
                                        angular-length ratio 0.5
  6. greedy best-first on normal alignment, one-to-one, never across cameras

Matched pairs get the SAME colour in both frames and a number. Unmatched
segments are drawn dim grey, so a low match rate is visible as a grey frame
rather than as a statistic.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))
from display import ROT180  # noqa: E402

# gates, straight from LineExtractor's constructor
GATE_NORMAL = np.deg2rad(3.0)
GATE_DIR = np.deg2rad(8.0)
GATE_LEN_RATIO = 0.5


def load_cam(path, cam):
    txt = Path(path).read_text()

    def g(k):
        m = re.search(rf"^{re.escape(k)}:\s*([-\d.eE+]+)", txt, re.M)
        return float(m.group(1)) if m else 0.0

    c = cam + 1
    return dict(fx=g(f"Camera{c}.fx"), fy=g(f"Camera{c}.fy"),
                cx=g(f"Camera{c}.cx"), cy=g(f"Camera{c}.cy"),
                d=[g(f"Camera{c}.k{i}") for i in (1, 2, 3, 4)],
                w=int(g("Camera.width")), h=int(g("Camera.height")))


def unproject_kb4(u, v, cam, iters=10):
    """pixel -> unit bearing, with a validity mask (corners fall outside)."""
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
    ok = np.isfinite(th) & (th < np.deg2rad(95.0))
    s = np.where(ru > 1e-9, np.sin(th) / np.maximum(ru, 1e-9), 1.0)
    b = np.stack([mx * s, my * s, np.cos(th)], -1)
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    return b / np.maximum(n, 1e-12), ok & (n[..., 0] > 1e-9)


def lift(segs, cam, mask, min_ang, max_ang):
    """segments -> (kept segments, normals, directions, angular lengths)."""
    if not len(segs):
        z = np.zeros((0, 3))
        return np.zeros((0, 4)), z, z, np.zeros(0)
    s = np.asarray(segs, float).reshape(-1, 4)
    if mask is not None:                     # step 2: rig hardware
        h, w = mask.shape
        def on(x, y):
            xi = np.clip(x.astype(int), 0, w - 1); yi = np.clip(y.astype(int), 0, h - 1)
            return mask[yi, xi] > 0
        s = s[~(on(s[:, 0], s[:, 1]) | on(s[:, 2], s[:, 3]))]
        if not len(s):
            z = np.zeros((0, 3)); return np.zeros((0, 4)), z, z, np.zeros(0)
    b1, v1 = unproject_kb4(s[:, 0], s[:, 1], cam)
    b2, v2 = unproject_kb4(s[:, 2], s[:, 3], cam)
    n = np.cross(b1, b2)
    ln = np.linalg.norm(n, axis=1)
    ang = np.arcsin(np.clip(ln, 0, 1))
    keep = v1 & v2 & (ln > 1e-9) & (ang >= min_ang) & (ang <= max_ang)
    s, n, ang = s[keep], n[keep] / ln[keep, None], ang[keep]
    d = b2[keep] - b1[keep]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return s, n, d, ang


def match(nc, dc, ac, np_, dp, ap):
    """The estimator's greedy best-first matcher, gates and all."""
    assign = np.full(len(nc), -1, int)
    if not len(nc) or not len(np_):
        return assign
    an = np.abs(nc @ np_.T)                       # sign-free: a line has no sense
    ad = np.abs(dc @ dp.T)
    lo = np.minimum(ac[:, None], ap[None, :])
    hi = np.maximum(ac[:, None], ap[None, :])
    ok = (an >= np.cos(GATE_NORMAL)) & (ad >= np.cos(GATE_DIR)) & \
         (lo / np.maximum(hi, 1e-12) >= GATE_LEN_RATIO)
    cand = [(an[i, j], i, j) for i, j in zip(*np.where(ok))]
    cand.sort(key=lambda c: -c[0])
    usedp = np.zeros(len(np_), bool)
    for _, i, j in cand:                          # one-to-one, best normal first
        if assign[i] == -1 and not usedp[j]:
            assign[i] = j; usedp[j] = True
    return assign


def run_elsed(tool, frames_dir, ids, tmp, grad, minlen):
    lst = tmp.with_suffix(".list")
    lst.write_text("\n".join(str(i) for i in ids) + "\n")
    subprocess.run([tool, str(frames_dir), str(lst), str(tmp), str(grad), str(minlen)],
                   capture_output=True, text=True, check=True)
    d = {}
    with open(tmp) as f:
        f.readline()
        for line in f:
            p = line.split(")")[0].split(",")
            d.setdefault(int(p[0]), []).append([float(x) for x in p[1:5]])
    lst.unlink(missing_ok=True); tmp.unlink(missing_ok=True)
    return {k: np.array(v) for k, v in d.items()}


def draw(img, segs, cols, scale):
    """Draw segments; return the canvas and each segment's midpoint in it."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if ROT180:
        vis = cv2.rotate(vis, cv2.ROTATE_180)
    H, W = vis.shape[:2]
    mids = []
    for k, (x1, y1, x2, y2) in enumerate(segs):
        if ROT180:
            x1, y1, x2, y2 = W - 1 - x1, H - 1 - y1, W - 1 - x2, H - 1 - y2
        col, matched = cols[k]
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), col,
                 3 if matched else 1, cv2.LINE_AA)
        mids.append(((x1 + x2) * 0.5 * scale, (y1 + y2) * 0.5 * scale))
    return cv2.resize(vis, None, fx=scale, fy=scale), mids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True, help="the ORB settings the run used")
    ap.add_argument("--elsed", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--pairs", required=True,
                    help="comma list of first frame ids; each pairs with --gap")
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.42)
    ap.add_argument("--min-ang", type=float, default=1.5)
    ap.add_argument("--max-ang", type=float, default=40.0)
    ap.add_argument("--grad", type=int, default=30)
    ap.add_argument("--minlen", type=int, default=15)
    ap.add_argument("--max-draw", type=int, default=45,
                    help="how many correspondences to DRAW (all are still "
                         "computed); 800 connectors is an unreadable hairball")
    ap.add_argument("--suspect-only", action="store_true",
                    help="draw only correspondences that move more than "
                         "--suspect-px, i.e. the ones most likely to be wrong")
    ap.add_argument("--suspect-px", type=float, default=60.0)
    a = ap.parse_args()

    ds = Path(a.dataset); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cams = [load_cam(a.config, c) for c in (0, 1)]
    masks = [None, None]
    if a.masks:
        for c in (0, 1):
            mp = Path(a.masks) / f"selfocc_cam{c}.png"
            if mp.exists():
                masks[c] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

    firsts = [int(x) for x in a.pairs.split(",")]
    ids = sorted({f for x in firsts for f in (x, x + a.gap)})
    segs = {c: run_elsed(a.elsed, ds / f"cam{c}", ids, out / f"_tmp{c}.csv",
                         a.grad, a.minlen) for c in (0, 1)}

    mn, mx = np.deg2rad(a.min_ang), np.deg2rad(a.max_ang)
    rng = np.random.default_rng(0)
    print(f"{'pair':>12s} {'cam':>4s} {'A':>6s} {'B':>6s} {'matched':>8s} {'rate':>7s}")
    for f0 in firsts:
        f1 = f0 + a.gap
        panes = []
        for c in (0, 1):
            imA = cv2.imread(str(ds / f"cam{c}" / f"{f0:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            imB = cv2.imread(str(ds / f"cam{c}" / f"{f1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if imA is None or imB is None:
                continue
            sA, nA, dA, aA = lift(segs[c].get(f0, []), cams[c], masks[c], mn, mx)
            sB, nB, dB, aB = lift(segs[c].get(f1, []), cams[c], masks[c], mn, mx)
            # B is "current", A is "previous" -- the estimator's direction
            asg = match(nB, dB, aB, nA, dA, aA)
            nm = int((asg >= 0).sum())
            rate = 100.0 * nm / max(len(nB), 1)
            print(f"{f0:6d}->{f1:<5d} {c:>4d} {len(sA):>6d} {len(sB):>6d} "
                  f"{nm:>8d} {rate:>6.1f}%")

            colA = [((80, 80, 80), False)] * len(sA)
            colA = list(colA)
            colB = []
            for i, j in enumerate(asg):
                if j < 0:
                    colB.append(((80, 80, 80), False))       # unmatched: dim grey
                else:
                    col = tuple(int(x) for x in rng.integers(70, 255, 3))
                    colB.append((col, True))
                    colA[j] = (col, True)
            pA, midA = draw(imA, sA, colA, a.scale)
            pB, midB = draw(imB, sB, colB, a.scale)
            lab = "FRONT" if c == 0 else "REAR"
            GAP = 90                       # room for the correspondence lines
            strip = np.hstack([pA, np.full((pA.shape[0], GAP, 3), 28, np.uint8), pB])
            # THE MATCHES THEMSELVES: one line per correspondence, drawn from the
            # segment's midpoint in A to its midpoint in B. A match that jumps to
            # a different part of the image is visible as a steeply sloped line,
            # which is what a wrong association looks like.
            xoff = pA.shape[1] + GAP
            pairs = [(i, j) for i, j in enumerate(asg) if j >= 0]
            if a.suspect_only:
                # A correspondence between consecutive frames should barely
                # move. The long jumps are where the matcher confused one line
                # for a parallel neighbour, so showing only those isolates the
                # failures instead of burying them in 800 correct ones.
                pairs = [(i, j) for i, j in pairs
                         if np.hypot(midB[i][0] - midA[j][0],
                                     midB[i][1] - midA[j][1]) > a.suspect_px * a.scale]
            if len(pairs) > a.max_draw:
                sel = rng.choice(len(pairs), a.max_draw, replace=False)
                pairs = [pairs[k] for k in sorted(sel)]
            ndrawn = len(pairs)
            for i, j in pairs:
                col = colB[i][0]
                x1, y1 = midA[j]
                x2, y2 = midB[i]
                cv2.line(strip, (int(x1), int(y1)), (int(x2 + xoff), int(y2)),
                         col, 1, cv2.LINE_AA)
                cv2.circle(strip, (int(x1), int(y1)), 3, col, -1, cv2.LINE_AA)
                cv2.circle(strip, (int(x2 + xoff), int(y2)), 3, col, -1, cv2.LINE_AA)
            cv2.putText(strip, f"{lab}  f{f0} -> f{f1}   {len(sA)} / {len(sB)} lines"
                              f"   matched {nm} ({rate:.0f}%)"
                              f"   showing {ndrawn}"
                              + ("  SUSPECT ONLY" if a.suspect_only else ""),
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panes.append(strip)
        if panes:
            w = max(p.shape[1] for p in panes)
            panes = [cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1],
                                        cv2.BORDER_CONSTANT, value=(30, 30, 30))
                     for p in panes]
            grid = np.vstack([panes[0], np.full((8, w, 3), 40, np.uint8), panes[1]]
                             if len(panes) > 1 else panes)
            p = out / f"match_f{f0:06d}_f{f1:06d}.png"
            cv2.imwrite(str(p), grid)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
