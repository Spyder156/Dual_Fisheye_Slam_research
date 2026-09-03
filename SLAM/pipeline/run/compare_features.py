#!/usr/bin/env python
"""ORB vs a learned extractor on the frames where tracking actually dies.

WHY: three detection-side knobs (CLAHE, lower FAST threshold, more features)
all made things WORSE -- each raised detections and lowered TRACKED features.
That says the bottleneck is match quality, not detection count. Learned
descriptors attack match quality directly, but integrating them into ORB-SLAM3
breaks DBoW2 (loop closure + relocalisation), so it is worth knowing the size of
the prize before paying that.

This matches consecutive frame pairs with each extractor and reports geometric
inliers, on two windows:
  - the BLACKOUT where both lenses drop to 0-2 tracked features
  - a healthy CONTROL window, so we know the harness itself is sane

Decision rule:
  learned finds real matches where ORB finds ~none -> integration is justified
  both fail                                        -> the frames carry no
                                                      recoverable information,
                                                      and the fix is IMU
                                                      bridging + map splitting
"""
import argparse
import subprocess
import sqlite3
import shutil
import os
from pathlib import Path

import numpy as np


def sh(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n"); f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def run_one(colmap, frames_dir, ids, out, extractor, cam_params, log, gpu="1"):
    """Extract + match consecutive pairs with one extractor; return inlier stats."""
    work = Path(out); shutil.rmtree(work, ignore_errors=True)
    (work / "imgs").mkdir(parents=True)
    names = []
    for i in ids:
        src = Path(frames_dir) / f"{i:06d}.jpg"
        if not src.exists():
            continue
        os.symlink(src.resolve(), work / "imgs" / src.name)
        names.append(src.name)
    db = work / "db.db"
    pairs = work / "pairs.txt"
    pairs.write_text("\n".join(f"{names[i]} {names[i+1]}"
                               for i in range(len(names) - 1)) + "\n")

    xopt = "FeatureExtraction"
    cmd = [colmap, "feature_extractor", "--database_path", db,
           "--image_path", work / "imgs",
           "--ImageReader.camera_model", "OPENCV_FISHEYE",
           "--ImageReader.single_camera", "1",
           "--ImageReader.camera_params", cam_params,
           f"--{xopt}.use_gpu", gpu]
    # COLMAP 4.x: FeatureExtraction.type selects SIFT / ALIKED / LOMA
    cmd += [f"--{xopt}.type", extractor.upper()]
    if sh(cmd, log):
        return None
    if sh([colmap, "matches_importer", "--database_path", db,
           "--match_list_path", pairs, "--match_type", "pairs",
           "--FeatureMatching.use_gpu", gpu], log):
        return None

    con = sqlite3.connect(str(db))
    kp = [r[0] for r in con.execute("select rows from keypoints")]
    raw = [r[0] for r in con.execute("select rows from matches")]
    ver = [r[0] for r in con.execute("select rows from two_view_geometries")]
    con.close()
    return dict(imgs=len(names), kp_med=float(np.median(kp)) if kp else 0,
                raw_med=float(np.median(raw)) if raw else 0,
                ver_med=float(np.median(ver)) if ver else 0,
                ver_zero=int(sum(1 for v in ver if v == 0)), pairs=len(ver))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--cam-params", required=True)
    ap.add_argument("--blackout", required=True, help="first:last frame id")
    ap.add_argument("--control", required=True, help="first:last frame id")
    ap.add_argument("--extractors", default="sift,aliked")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = out / "colmap.log"
    wins = {}
    for tag, spec in (("BLACKOUT", a.blackout), ("control", a.control)):
        lo, hi = (int(x) for x in spec.split(":"))
        wins[tag] = list(range(lo, hi + 1))

    print(f"{'window':10s} {'extractor':10s} {'imgs':>5s} {'kp/img':>8s} "
          f"{'raw':>8s} {'verified':>9s} {'0-inlier pairs':>15s}")
    for tag, ids in wins.items():
        for ex in a.extractors.split(","):
            r = run_one(a.colmap, a.frames_dir, ids, out / f"{tag}_{ex}",
                        ex, a.cam_params, log)
            if r is None:
                print(f"{tag:10s} {ex:10s}  FAILED (see {log})")
                continue
            print(f"{tag:10s} {ex:10s} {r['imgs']:5d} {r['kp_med']:8.0f} "
                  f"{r['raw_med']:8.0f} {r['ver_med']:9.0f} "
                  f"{r['ver_zero']:6d}/{r['pairs']:<8d}")


if __name__ == "__main__":
    main()
