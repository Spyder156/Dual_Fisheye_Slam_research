#!/usr/bin/env python
"""Text-prompted person masks for COLMAP (SAM3 semantic, per-frame).

For each staged image: segment PROMPTS with SAM3, dilate, and subtract from the
circular fisheye validity mask -> per-image COLMAP mask PNG (255 = usable,
0 = ignored: outside circle OR on the camera-holder).

Usage:
  gen_person_masks.py <images_dir> <masks_out_dir> --circle CX CY R [--model sam3.pt]
Writes <masks_out_dir>/<name>.jpg.png per image, plus preview overlays for the
first few frames in <masks_out_dir>/../mask_preview/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

PROMPTS = ["person", "hand", "arm", "held object", "phone"]
CONF = 0.25
DILATE = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--circle", type=float, nargs=3, required=True, metavar=("CX", "CY", "R"))
    ap.add_argument("--model", default="/home/raghav/workspace/MeckaAI/Project1/models/sam3.pt")
    ap.add_argument("--infer-size", type=int, default=1024)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prev_dir = args.out.parent / "mask_preview"
    prev_dir.mkdir(exist_ok=True)

    from ultralytics.models.sam import SAM3SemanticPredictor
    predictor = SAM3SemanticPredictor(
        overrides=dict(conf=CONF, task="segment", mode="predict", model=args.model,
                       half=True, save=False, verbose=False, device="cuda"))

    files = sorted(args.images.glob("*.jpg"))
    cx, cy, r = args.circle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))
    fracs = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        s = args.infer_size / max(H, W)
        small = cv2.resize(img, None, fx=s, fy=s)
        predictor.set_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        results = predictor(text=PROMPTS)
        person = np.zeros(small.shape[:2], np.uint8)
        if results and results[0].masks is not None and len(results[0].masks.data):
            person = (results[0].masks.data.cpu().numpy() > 0.5).any(0).astype(np.uint8)
            if person.shape != small.shape[:2]:
                person = cv2.resize(person, (small.shape[1], small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        person = cv2.dilate(person, kernel, iterations=2)
        person = cv2.resize(person, (W, H), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.mgrid[0:H, 0:W]
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8) * 255
        mask[person > 0] = 0
        cv2.imwrite(str(args.out / (f.name + ".png")), mask)
        fracs.append((person > 0).mean())
        if i < 6 or i % 100 == 0:
            ov = img.copy()
            ov[person > 0] = (0.4 * ov[person > 0] + np.array([0, 0, 153])).astype(np.uint8)
            cv2.circle(ov, (int(cx), int(cy)), int(r), (0, 255, 0), 4)
            cv2.imwrite(str(prev_dir / f"prev_{f.stem}.jpg"),
                        cv2.resize(ov, None, fx=0.25, fy=0.25))
        if i % 50 == 0:
            print(f"{i}/{len(files)} person-frac mean {np.mean(fracs)*100:.1f}%", flush=True)
    print(f"done: {len(files)} masks, person coverage mean {np.mean(fracs)*100:.1f}% "
          f"max {np.max(fracs)*100:.1f}% -> {args.out}")


if __name__ == "__main__":
    main()
