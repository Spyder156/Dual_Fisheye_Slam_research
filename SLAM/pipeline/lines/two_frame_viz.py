#!/usr/bin/env python
"""Draw the two_frame_match output.

viz1: every observation the pipeline would MATCH ON (post-mask, post-merge),
      side by side. What the matcher sees, nothing else.
viz2: matched pairs in the same colour with the same number in both frames;
      unmatched observations dim grey. A weak matcher shows up as a grey image.

Observations are drawn as their great-circle ARCS projected through the
calibrated model (merged observations have no meaningful raw pixel ends).
Displayed rotated 180 deg -- the sensor is mounted upside down.

usage: two_frame_viz.py <prefix> <imgA> <imgB> --config <yaml> [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from debug_line_projection import load_cam  # noqa: E402
from doorway_overlay import project_kb4     # noqa: E402


def arc_px(row, cam, n=48):
    b1 = np.array([row["b1x"], row["b1y"], row["b1z"]])
    b2 = np.array([row["b2x"], row["b2y"], row["b2z"]])
    th = np.arccos(np.clip(b1 @ b2, -1, 1))
    if th < 1e-6:
        return None
    t = np.linspace(0, 1, n)[:, None]
    B = (np.sin((1 - t) * th) * b1 + np.sin(t * th) * b2) / np.sin(th)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return project_kb4(B, cam).astype(np.int32)


def draw(img, rows, cam, color_of, label_of):
    # rotate FIRST, then draw with flipped coordinates -- labels drawn before
    # the rotation come out upside down
    out = cv2.rotate(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.ROTATE_180)
    h, w = out.shape[:2]
    for r in rows:
        uv = arc_px(r, cam)
        if uv is None:
            continue
        uv = np.stack([w - 1 - uv[:, 0], h - 1 - uv[:, 1]], -1)
        c, lbl = color_of(int(r["idx"])), label_of(int(r["idx"]))
        cv2.polylines(out, [uv.reshape(-1, 1, 2)], False, c, 2, cv2.LINE_AA)
        if lbl is not None:
            cv2.putText(out, str(lbl), tuple(uv[len(uv)//2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("imgA"); ap.add_argument("imgB")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(args.prefix).parent
    cam = load_cam(args.config, 0)

    A = np.atleast_1d(np.genfromtxt(f"{args.prefix}_segA.csv", delimiter=",", names=True))
    B = np.atleast_1d(np.genfromtxt(f"{args.prefix}_segB.csv", delimiter=",", names=True))
    M = np.atleast_1d(np.genfromtxt(f"{args.prefix}_match.csv", delimiter=",", names=True))
    imA = cv2.imread(args.imgA, cv2.IMREAD_GRAYSCALE)
    imB = cv2.imread(args.imgB, cv2.IMREAD_GRAYSCALE)

    # viz1: everything, plain yellow
    y = (0, 220, 255)
    v1 = np.hstack([draw(imA, A, cam, lambda i: y, lambda i: None),
                    draw(imB, B, cam, lambda i: y, lambda i: None)])
    cv2.imwrite(str(out / "viz1_detected.png"), v1)

    # viz2: matched pairs same colour+number, unmatched grey
    pair_of_A, pair_of_B = {}, {}
    for k, m in enumerate(M):
        pair_of_B[int(m["iB"])] = k
        pair_of_A[int(m["iA"])] = k
    rng = np.random.default_rng(3)
    cols = [tuple(int(c) for c in rng.integers(70, 255, 3)) for _ in range(len(M) + 1)]
    grey = (90, 90, 90)
    v2 = np.hstack([
        draw(imA, A, cam,
             lambda i: cols[pair_of_A[i]] if i in pair_of_A else grey,
             lambda i: pair_of_A.get(i)),
        draw(imB, B, cam,
             lambda i: cols[pair_of_B[i]] if i in pair_of_B else grey,
             lambda i: pair_of_B.get(i))])
    cv2.imwrite(str(out / "viz2_matched.png"), v2)
    print(f"A {len(A)} obs, B {len(B)} obs, {len(M)} matches "
          f"-> {out}/viz1_detected.png, viz2_matched.png")


if __name__ == "__main__":
    main()
