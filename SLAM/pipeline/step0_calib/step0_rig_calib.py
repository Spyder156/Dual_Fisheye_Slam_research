#!/usr/bin/env python
"""Step 0: per-unit intrinsics + rig-extrinsics refinement from images alone.

WHY: everyone assumes a dual-fisheye rig is exactly back-to-back (180.0 deg) with
a negligible baseline. Neither is true. Hilti's own calibration says 179.56 deg
and 4.01 cm. For an Insta360 there is no such calibration at all -- recovering it
per unit is the point of the project.

THE PROBLEM: front and back never see the same scene at the same instant, so a
naive COLMAP run yields two disconnected sub-models and the rig extrinsics are
unrecoverable.

THE TRICK (no SLAM in the loop): walking forward, the REAR camera at t+dt looks
back at what the FRONT camera saw at t. So we match front[i] <-> back[i+dt].
dt depends on scene depth (near features cross quickly, far ones slowly), so we
use a SET of offsets rather than one.

Pair list built here:
  front[i] <-> front[j]   for |i-j| <= WINDOW      (intrinsics, cam0)
  back[i]  <-> back[j]    for |i-j| <= WINDOW      (intrinsics, cam1)
  front[i] <-> back[i+dt] for dt in OFFSETS        (the rig link)

Usage:
  step0_rig_calib.py --dataset Data/Hilti/ds/<seq> --out Data/Hilti/step0/<seq> \
      [--stride 15] [--window 4] [--offsets 5,10,20,40] [--max-frames 400]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

COLMAP = "/cb/src/colmap/exe/colmap"
LD = "/cb/_deps/onnxruntime-build/lib"      # COLMAP links a vendored onnxruntime


def docker(repo, args, gpu=False):
    """Run a colmap subcommand inside the fisheye-slam image."""
    cmd = ["docker", "run", "--rm", "-u", f"{Path.home().stat().st_uid}:{Path.home().stat().st_gid}",
           "-e", "HOME=/tmp", "-e", f"LD_LIBRARY_PATH={LD}"]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += ["-v", f"{repo}/SLAM/build/colmap_build:/cb",
            "-v", f"{repo}/Data:/data",
            "fisheye-slam", COLMAP] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def build_pairs(n_front, n_back, window, offsets, names_f, names_b):
    pairs = []
    for i in range(n_front):                       # cam0 intra -> intrinsics
        for j in range(i + 1, min(i + window + 1, n_front)):
            pairs.append((names_f[i], names_f[j]))
    for i in range(n_back):                        # cam1 intra -> intrinsics
        for j in range(i + 1, min(i + window + 1, n_back)):
            pairs.append((names_b[i], names_b[j]))
    n_cross = 0
    for dt in offsets:                             # THE RIG LINK
        for i in range(n_front):
            j = i + dt
            if 0 <= j < n_back:
                pairs.append((names_f[i], names_b[j]))
                n_cross += 1
    return pairs, n_cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=15, help="keep every Nth frame")
    ap.add_argument("--window", type=int, default=4, help="intra-camera pair window")
    ap.add_argument("--offsets", default="5,10,20,40",
                    help="front[i]<->back[i+dt] offsets, in SELECTED-frame units")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--gpu", action="store_true", help="GPU SIFT + matching")
    ap.add_argument("--camera-model", default="OPENCV_FISHEYE")
    ap.add_argument("--camera-params", default=None,
                    help="fx,fy,cx,cy,k1,k2,k3,k4 seed. WITHOUT this COLMAP guesses "
                         "focal ~1.2*max_dim (~1766 px here) vs the true ~465, and "
                         "EVERY two-view geometry fails verification -- 0 inliers.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    offsets = [int(x) for x in args.offsets.split(",")]
    out = args.out
    img = out / "images"
    (img / "cam0").mkdir(parents=True, exist_ok=True)
    (img / "cam1").mkdir(parents=True, exist_ok=True)

    # ---- select a sparse subset; full framerate is pointless for calibration ----
    frames = np.genfromtxt(args.dataset / "frames.csv", delimiter=",", names=True)
    fids = np.atleast_1d(frames["frame"]).astype(int)[::args.stride][:args.max_frames]
    names_f, names_b = [], []
    for k in fids:
        for cam, names in (("cam0", names_f), ("cam1", names_b)):
            src = args.dataset / cam / f"{k:06d}.jpg"
            if src.exists():
                dst = img / cam / f"{k:06d}.jpg"
                if not dst.exists():
                    shutil.copy(src, dst)
                names.append(f"{cam}/{k:06d}.jpg")
    print(f"selected {len(names_f)} front + {len(names_b)} back frames "
          f"(stride {args.stride} of {len(np.atleast_1d(frames['frame']))})")

    pairs, n_cross = build_pairs(len(names_f), len(names_b), args.window, offsets,
                                 names_f, names_b)
    pair_file = out / "pairs.txt"
    pair_file.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n")
    print(f"pair list: {len(pairs)} pairs ({n_cross} cross-camera, the rig link)")

    d = f"/data/{out.resolve().relative_to((repo/'Data').resolve())}"
    db = f"{d}/database.db"

    # ---- feature extraction: one camera per folder, so cam0/cam1 stay distinct ----
    print("feature extraction...")
    r = docker(repo, ["feature_extractor", "--database_path", db,
                      "--image_path", f"{d}/images",
                      "--ImageReader.camera_model", args.camera_model,
                      "--ImageReader.single_camera_per_folder", "1",
                      "--FeatureExtraction.use_gpu", "1" if args.gpu else "0"]
                     + (["--ImageReader.camera_params", args.camera_params]
                        if args.camera_params else []), gpu=args.gpu)
    if r.returncode:
        print(r.stderr[-1500:]); return

    print("matching the custom pair list...")
    r = docker(repo, ["matches_importer", "--database_path", db,
                      "--match_list_path", f"{d}/pairs.txt",
                      "--match_type", "pairs",
                      "--FeatureMatching.use_gpu", "1" if args.gpu else "0"], gpu=args.gpu)
    if r.returncode:
        print(r.stderr[-1500:]); return

    print("mapping...")
    (out / "sparse").mkdir(exist_ok=True)
    r = docker(repo, ["mapper", "--database_path", db,
                      "--image_path", f"{d}/images",
                      "--output_path", f"{d}/sparse",
                      # principal point off by default; we WANT it refined
                      "--Mapper.ba_refine_principal_point", "1"])
    if r.returncode:
        print(r.stderr[-1500:]); return
    print(r.stdout[-800:])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
