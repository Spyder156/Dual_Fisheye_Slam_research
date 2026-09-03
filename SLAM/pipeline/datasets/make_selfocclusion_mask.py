#!/usr/bin/env python
"""Build a self-occlusion mask from the footage itself.

WHY. The rig's own hardware sits in the bottom of the rear lens in every frame.
ELSED and ORB both find strong features on it -- in the blackout at frame 4500
almost every rear line is on the device, not the scene.

Those features are not merely useless, they are HARMFUL: the hardware is static
in the camera frame, so its features say "no motion" while the rig is moving.
Zero features from a lens is a clean signal that the other lens must carry;
static features are a wrong signal that fights the correct one.

HOW. Self-occlusion is precisely the region that never changes. Sample frames
across the whole sequence and take the per-pixel temporal standard deviation of
intensity: scene pixels vary as the camera moves, hardware pixels do not.

The threshold is chosen on the DISTRIBUTION, not a magic constant, and the
script refuses to emit an implausibly large mask -- a mask that swallows the
scene would silently destroy tracking, which is a far worse failure than no mask
at all.

Output follows the convention already used across this project:
NONZERO == MASKED (ignore). Note this is the opposite of what OKVIS2's docs
claim and the same as what its code actually does.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))
from display import show  # noqa: E402


def build(frames_dir, ids, std_pct, min_keep=0.55, max_mask=0.35):
    acc = None
    acc2 = None
    n = 0
    for i in ids:
        img = cv2.imread(str(Path(frames_dir) / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        f = img.astype(np.float32)
        acc = f if acc is None else acc + f
        acc2 = f * f if acc2 is None else acc2 + f * f
        n += 1
    if n < 10:
        raise SystemExit(f"only {n} frames readable in {frames_dir}")
    mean = acc / n
    var = np.maximum(acc2 / n - mean * mean, 0)
    std = np.sqrt(var)

    thr = np.percentile(std, std_pct)
    mask = (std < thr).astype(np.uint8) * 255

    # Clean up: the hardware is one solid blob, not speckle.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # Keep only components touching the image border: self-occlusion is
    # attached to the edge of the view (the body of the device), whereas a dark
    # static patch in the middle of the scene is a wall, and masking walls is
    # how you destroy tracking.
    num, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    H, W = mask.shape
    for c in range(1, num):
        x, y, w, h, area = stats[c]
        touches = (x <= 1) or (y <= 1) or (x + w >= W - 1) or (y + h >= H - 1)
        if touches and area > 0.002 * H * W:
            keep[lab == c] = 255
    mask = keep

    frac = float((mask > 0).mean())
    return mask, std, frac, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--std-pct", type=float, default=12.0,
                    help="pixels below this percentile of temporal std are static")
    ap.add_argument("--max-mask", type=float, default=0.35,
                    help="refuse to emit a mask covering more than this fraction")
    a = ap.parse_args()

    ds = Path(a.dataset)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int)
    ids = fid[np.linspace(0, len(fid) - 1, min(a.samples, len(fid))).astype(int)]

    for c in (0, 1):
        mask, std, frac, thr = build(ds / f"cam{c}", ids, a.std_pct)
        print(f"cam{c}: std threshold {thr:6.2f}  masked {100*frac:5.2f}% of the image")
        if frac > a.max_mask:
            print(f"  REFUSING: {100*frac:.1f}% is implausible for self-occlusion. "
                  f"A mask that eats the scene is worse than no mask.")
            continue
        cv2.imwrite(str(out / f"selfocc_cam{c}.png"), mask)

        # visual proof: the mask drawn over a real frame, plus the std map that
        # produced it. Never ship a mask you have not looked at.
        img = cv2.imread(str(ds / f"cam{c}" / f"{ids[len(ids)//2]:06d}.jpg"),
                         cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(show(img), cv2.COLOR_GRAY2BGR)
        mask_d = show(mask)
        red = np.zeros_like(vis); red[:, :, 2] = 255
        m3 = (mask_d > 0)[:, :, None]
        vis = np.where(m3, (0.45 * vis + 0.55 * red).astype(np.uint8), vis)
        sv = cv2.cvtColor(show((np.clip(std, 0, 60) / 60 * 255).astype(np.uint8)),
                          cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, f"cam{c} self-occlusion  {100*frac:.1f}% masked",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(sv, "temporal std (dark = static)", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        pair = np.hstack([vis, np.full((vis.shape[0], 8, 3), 30, np.uint8), sv])
        cv2.imwrite(str(out / f"selfocc_cam{c}_check.png"),
                    cv2.resize(pair, None, fx=0.5, fy=0.5))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
