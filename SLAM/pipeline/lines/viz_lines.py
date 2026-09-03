#!/usr/bin/env python
"""Visual output for the line/point front end. Images, not charts.

Every panel is a 2x2 grid of the actual frames:

        frame A  cam0 (FRONT) | frame A  cam1 (REAR)
        frame B  cam0 (FRONT) | frame B  cam1 (REAR)

A feature matched between A and B is drawn in the SAME colour in both rows, so a
correct match is visible as a colour appearing twice in the same place on the
structure. Unmatched features are dim grey -- they are context, not evidence.

Folders produced:
  lines/         cross-frame LINE matches
  points/        cross-frame POINT matches, same layout, for direct comparison
  blackout/      the frames where point tracking collapses and lines do not
  rig_view/      the two lenses at the SAME instant -- they must share nothing
  triangulation/ which lines have enough baseline to become landmarks
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend import KLTPoints, SphericalLines, load_cam, run_elsed  # noqa: E402

FRONT_TAG, REAR_TAG = "FRONT (cam0)", "REAR (cam1)"


def colour(i):
    """Stable, well-separated colour per match index."""
    h = int((i * 47) % 180)
    return tuple(int(x) for x in
                 cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


# The Hilti sensor is mounted inverted: world-up projects onto image +y (down)
# on BOTH lenses. The calibration encodes that, so the estimator is correct on
# raw pixels -- only the DISPLAY is upside down to a human. Rotate the image and
# the drawn coordinates together, never the geometry.
ROT180 = True


def to_bgr(img, scale=0.5):
    b = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if ROT180:
        b = cv2.rotate(b, cv2.ROTATE_180)
    return cv2.resize(b, None, fx=scale, fy=scale)


def xy(px, py, cam, scale):
    """pixel -> display coords, applying the same 180 deg the image got."""
    if ROT180:
        px = cam["w"] - 1 - px
        py = cam["h"] - 1 - py
    return int(px * scale), int(py * scale)


def label(img, txt, extra=""):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if extra:
        cv2.putText(img, extra, (img.shape[1] - 230, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (120, 220, 255), 1)
    return img


def grid(tiles, gap=6):
    (a, b), (c, d) = tiles
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    out = np.full((2 * h + gap, 2 * w + gap, 3), 30, np.uint8)
    out[:h, :w] = a; out[:h, w + gap:w + gap + b.shape[1]] = b
    out[h + gap:h + gap + c.shape[0], :w] = c
    out[h + gap:h + gap + d.shape[0], w + gap:w + gap + d.shape[1]] = d
    return out


def draw_lines(img, obs, assign_colour, scale, cam):
    """assign_colour: index -> colour or None (unmatched)."""
    for i, o in enumerate(obs):
        c = assign_colour(i)
        p1 = xy(o["p1"][0], o["p1"][1], cam, scale)
        p2 = xy(o["p2"][0], o["p2"][1], cam, scale)
        if c is None:
            cv2.line(img, p1, p2, (90, 90, 90), 1, cv2.LINE_AA)
        else:
            cv2.line(img, p1, p2, c, 2, cv2.LINE_AA)
    return img


def segs_to_obs(segs, sl):
    """pixel segments -> list of dicts carrying what the drawing needs."""
    if segs is None or len(segs) == 0:
        return []
    n, d, ang = sl.lift(segs)
    # sl.lift filters, so re-derive the surviving pixel segments the same way
    s = np.asarray(segs, float).reshape(-1, 4)
    from frontend import unproject_kb4
    b1, v1 = unproject_kb4(s[:, 0], s[:, 1], sl.cam)
    b2, v2 = unproject_kb4(s[:, 2], s[:, 3], sl.cam)
    nn = np.cross(b1, b2); ln = np.linalg.norm(nn, axis=1)
    with np.errstate(invalid="ignore"):
        a = np.arcsin(np.clip(ln, 0, 1))
    ok = v1 & v2 & (ln > 1e-9) & (a >= sl.min_ang)
    out = []
    for k in np.where(ok)[0]:
        out.append(dict(p1=s[k, :2], p2=s[k, 2:4],
                        n=nn[k] / ln[k], ang=a[k]))
    return out


def match_obs(cur, prv, sl):
    """Three-gate matching on the dict form."""
    if not cur or not prv:
        return {}
    N = np.stack([o["n"] for o in prv]); A = np.array([o["ang"] for o in prv])
    M = np.stack([o["n"] for o in cur]); B = np.array([o["ang"] for o in cur])
    cn = np.abs(N @ M.T)
    lr = np.minimum(A[:, None], B[None, :]) / np.maximum(
        np.maximum(A[:, None], B[None, :]), 1e-12)
    ok = (cn >= np.cos(sl.g_normal)) & (lr >= sl.g_len)
    score = np.where(ok, cn, -1.0)
    out, usedc = {}, set()
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    usedp = set()
    for pi, ci in order:
        if score[pi, ci] < 0:
            break
        if pi in usedp or ci in usedc:
            continue
        usedp.add(pi); usedc.add(ci); out[int(ci)] = int(pi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--imucam", required=True)
    ap.add_argument("--elsed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None,
                    help="comma list of 'A:B' frame ids; default picks a spread")
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--scale", type=float, default=0.45)
    a = ap.parse_args()

    ds = Path(a.dataset); out = Path(a.out)
    for sub in ("lines", "points", "blackout", "rig_view", "triangulation"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cams = [load_cam(a.imucam, 0), load_cam(a.imucam, 1)]
    sls = [SphericalLines(cams[0]), SphericalLines(cams[1])]

    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int)

    if a.pairs:
        pairs = [tuple(int(x) for x in p.split(":")) for p in a.pairs.split(",")]
    else:
        pick = np.linspace(fid[10], fid[-10], 8).astype(int)
        pairs = [(int(p), int(p) + a.gap) for p in pick]
    # always include the measured blackout
    blackout = [(4700, 4710), (4720, 4730), (4740, 4750)]

    need = sorted({f for p in pairs + blackout for f in p})
    segs = {}
    for c in (0, 1):
        segs[c] = run_elsed(a.elsed, ds / f"cam{c}", need,
                            out / f"_segs_cam{c}.csv")

    def build(A, B, folder, tag):
        tiles = []
        for f in (A, B):
            row = []
            for c in (0, 1):
                img = cv2.imread(str(ds / f"cam{c}" / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return False
                row.append(to_bgr(img, a.scale))
            tiles.append(row)
        stats = []
        for c in (0, 1):
            oa = segs_to_obs(segs[c].get(A, np.zeros((0, 4))), sls[c])
            ob = segs_to_obs(segs[c].get(B, np.zeros((0, 4))), sls[c])
            m = match_obs(ob, oa, sls[c])          # cur(B) -> prev(A)
            inv = {v: k for k, v in m.items()}
            draw_lines(tiles[0][c], oa, lambda i: colour(inv[i]) if i in inv else None, a.scale, cams[c])
            draw_lines(tiles[1][c], ob, lambda i: colour(m[i]) if i in m else None, a.scale, cams[c])
            stats.append((len(oa), len(ob), len(m)))
        label(tiles[0][0], f"frame {A}  {FRONT_TAG}", f"{stats[0][0]} lines")
        label(tiles[0][1], f"frame {A}  {REAR_TAG}", f"{stats[1][0]} lines")
        label(tiles[1][0], f"frame {B}  {FRONT_TAG}", f"{stats[0][2]} matched")
        label(tiles[1][1], f"frame {B}  {REAR_TAG}", f"{stats[1][2]} matched")
        cv2.imwrite(str(folder / f"{tag}_{A:06d}_{B:06d}.png"), grid(tiles))
        return True

    for A, B in pairs:
        build(A, B, out / "lines", "lines")
    for A, B in blackout:
        build(A, B, out / "blackout", "BLACKOUT_lines")
    print(f"lines     -> {len(list((out/'lines').glob('*.png')))} images")
    print(f"blackout  -> {len(list((out/'blackout').glob('*.png')))} images")

    # ---- points, same layout, for direct comparison -------------------------
    for A, B in pairs + blackout:
        tiles = []
        ok = True
        for f in (A, B):
            row = []
            for c in (0, 1):
                img = cv2.imread(str(ds / f"cam{c}" / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    ok = False; break
                row.append(to_bgr(img, a.scale))
            if not ok:
                break
            tiles.append(row)
        if not ok:
            continue
        for c in (0, 1):
            ia = cv2.imread(str(ds / f"cam{c}" / f"{A:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            ib = cv2.imread(str(ds / f"cam{c}" / f"{B:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            p0 = cv2.goodFeaturesToTrack(ia, 900, 0.01, 12)
            if p0 is None:
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(ia, ib, p0, None,
                                                 winSize=(21, 21), maxLevel=4)
            st = st.ravel().astype(bool)
            n = 0
            for k, (q0, q1) in enumerate(zip(p0.reshape(-1, 2), p1.reshape(-1, 2))):
                if not st[k]:
                    cv2.circle(tiles[0][c], xy(q0[0], q0[1], cams[c], a.scale),
                               2, (90, 90, 90), -1)
                    continue
                col = colour(k)
                cv2.circle(tiles[0][c], xy(q0[0], q0[1], cams[c], a.scale), 3, col, -1)
                cv2.circle(tiles[1][c], xy(q1[0], q1[1], cams[c], a.scale), 3, col, -1)
                n += 1
            label(tiles[0][c], f"frame {A}  cam{c}", f"{len(p0)} pts")
            label(tiles[1][c], f"frame {B}  cam{c}", f"{n} tracked")
        cv2.imwrite(str(out / "points" / f"points_{A:06d}_{B:06d}.png"), grid(tiles))
    print(f"points    -> {len(list((out/'points').glob('*.png')))} images")

    # ---- rig view: the two lenses at the SAME instant ----------------------
    # If the rig is right these two images share NO structure. Any apparent
    # overlap would mean the extrinsics are wrong.
    for f in [p[0] for p in pairs][:6]:
        row = []
        for c in (0, 1):
            img = cv2.imread(str(ds / f"cam{c}" / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                break
            t = to_bgr(img, a.scale)
            o = segs_to_obs(segs[c].get(f, np.zeros((0, 4))), sls[c])
            draw_lines(t, o, lambda i: (60, 130, 255) if c == 0 else (70, 70, 255), a.scale, cams[c])
            label(t, f"frame {f}  {'FRONT' if c==0 else 'REAR'}", f"{len(o)} lines")
            row.append(t)
        if len(row) == 2:
            cv2.imwrite(str(out / "rig_view" / f"rig_{f:06d}.png"),
                        np.hstack([row[0], np.full((row[0].shape[0], 6, 3), 30, np.uint8), row[1]]))
    print(f"rig_view  -> {len(list((out/'rig_view').glob('*.png')))} images")


if __name__ == "__main__":
    main()
