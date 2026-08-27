#!/usr/bin/env python3
"""Build a COLMAP model from a trajectory whose poses we already trust.

Why this exists: the refinement BA is the single largest gain in the pipeline
(+11 to +17 QA points), but it needs a model -- poses AND points AND the
observations linking them. A VIO front-end like OpenVINS gives us excellent
poses and no structure, so without this step the best front-end in the tree is
the one branch that never gets refined. That is backwards.

So: hold the VIO poses FIXED, detect and match features, and triangulate
structure into them. COLMAP's point_triangulator does exactly this. The poses do
not move here -- they are the input, not the output. Moving them is the BA's job
in the next stage, where the IMU gets a vote.

Frame conventions, which are the easy thing to get wrong:
  input TUM   world-from-BODY (IMU), the frame everything downstream uses
  COLMAP      cam-from-world
so each pose is converted through the camera<-IMU extrinsic before writing.

Usage: triangulate_from_poses.py <episode> <traj.tum> <frames_dir>
                                 <calib.json> <out_model> [--matcher seq|exh]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

P1 = Path(os.environ.get("MECKA_ROOT", "/pipeline/data"))
# camera -> IMU(body); identical to the aligner and the C++ solver's T_IMU_CAM
R_BC = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
T_BC = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])


def sh(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} "
                           f"failed (exit {r.returncode}); see {log}")


def colmap_opt_prefixes():
    """('FeatureExtraction', 'FeatureMatching') on COLMAP 4.x, the Sift* names
    on 3.x.

    COLMAP moved the options that are not SIFT-specific onto their own prefix
    between 3.x and 4.x. An unrecognised option is not a hard failure -- colmap
    logs it and proceeds with the DEFAULT value -- so passing 3.x names to a 4.x
    binary silently ignores use_gpu and max_image_size, and the damage only shows
    up much later as an empty model. Probed rather than assumed so this works
    against whichever colmap is on PATH.
    """
    try:
        h = subprocess.run(["colmap", "feature_extractor", "--help"],
                           capture_output=True, text=True, timeout=60)
        out = (h.stdout or "") + (h.stderr or "")
        if "FeatureExtraction.max_image_size" in out:
            return "FeatureExtraction", "FeatureMatching"
    except (OSError, subprocess.SubprocessError):
        pass
    return "SiftExtraction", "SiftMatching"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("traj")
    ap.add_argument("frames_dir")
    ap.add_argument("calib")
    ap.add_argument("out_model")
    ap.add_argument("--matcher", default="seq", choices=["seq", "exh"])
    ap.add_argument("--max-frames", type=int, default=400,
                    help="cap registered images; triangulation cost is "
                         "superlinear and 400 well-spread views is plenty")
    a = ap.parse_args()

    frames = Path(a.frames_dir)
    out = Path(a.out_model)
    work = out.parent / (out.name + "_work")
    work.mkdir(parents=True, exist_ok=True)
    log = work / "colmap.log"
    cal = json.loads(Path(a.calib).read_text())
    fx, fy, cx, cy = cal["full_res_params"][:4]
    dist = ([*list(cal["full_res_params"][4:8]), 0, 0, 0, 0])[:4]

    # ---- frame index -> unix time, on the corrected clock -------------------
    ep_dir = P1 / "data/episodes" / a.episode
    fcsv = sorted(ep_dir.glob("frames*.csv"))[0]
    off = json.loads((P1 / "results/offsets" / f"{a.episode}.json").read_text())
    unix = pd.read_csv(fcsv)["unix_timestamp"].to_numpy() + off["offset_s"]

    traj = np.loadtxt(a.traj, comments="#")
    t_tr = traj[:, 0]

    imgs = sorted(p for p in frames.iterdir() if p.suffix.lower() in
                  (".jpg", ".jpeg", ".png"))
    if not imgs:
        raise SystemExit(f"no frames in {frames}")
    if len(imgs) > a.max_frames:
        keep = np.linspace(0, len(imgs) - 1, a.max_frames).astype(int)
        imgs = [imgs[i] for i in keep]

    # ---- a model with known poses and no points -----------------------------
    pose_dir = work / "known_poses"
    pose_dir.mkdir(exist_ok=True)
    (pose_dir / "points3D.txt").write_text("")
    W, H = cal.get("width", 1920), cal.get("height", 1080)
    (pose_dir / "cameras.txt").write_text(
        f"1 OPENCV {W} {H} {fx} {fy} {cx} {cy} "
        + " ".join(str(d) for d in dist) + "\n")

    # Pose per FILENAME first. The COLMAP image_id cannot be chosen here: the
    # database assigns its own when feature_extractor indexes the files, and
    # point_triangulator matches the two by ID, not by name. Writing our own
    # sequential ids produced
    #   Check failed: existing_image.Name() == image.second.Name()
    #                 (003678.jpg vs. 002328.jpg)
    # because id 1 meant a different file to each side. So the ids get filled
    # in after the database exists, from the database.
    pose_by_name = {}
    for p in imgs:
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        if idx >= len(unix):
            continue
        k = int(np.abs(t_tr - unix[idx]).argmin())
        # a pose more than half a frame away is an extrapolation, not a pose
        if abs(t_tr[k] - unix[idx]) > 0.02:
            continue
        p_wb = traj[k, 1:4]
        R_wb = Rotation.from_quat(traj[k, 4:8]).as_matrix()
        R_wc = R_wb @ R_BC                      # body -> camera
        p_wc = p_wb + R_wb @ T_BC
        R_cw = R_wc.T
        t_cw = -R_cw @ p_wc
        q = Rotation.from_matrix(R_cw).as_quat()   # xyzw
        pose_by_name[p.name] = (f"{q[3]} {q[0]} {q[1]} {q[2]} "
                                f"{t_cw[0]} {t_cw[1]} {t_cw[2]}")
    if len(pose_by_name) < 20:
        raise SystemExit(f"only {len(pose_by_name)} frames had a pose within "
                         f"20 ms -- the trajectory and the frame clock disagree")

    # ---- features and matches ----------------------------------------------
    db = work / "database.db"
    if db.exists():
        db.unlink()
    xopt, mopt = colmap_opt_prefixes()
    # GPU SIFT. This was hardcoded to CPU because the PACKAGED colmap was built
    # without CUDA and aborted headless trying to make an OpenGL context. The
    # image now ships a CUDA build, so that no longer applies -- and on CPU this
    # stage ran 6+ minutes per episode with the card sitting at 0%, on up to 400
    # full-resolution frames. Override with MECKA_TRIANGULATE_GPU=0.
    gpu = os.environ.get("MECKA_TRIANGULATE_GPU", "1")
    sh(["colmap", "feature_extractor", "--database_path", db,
        "--image_path", frames, "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy},"
        + ",".join(str(d) for d in dist),
        f"--{xopt}.use_gpu", gpu], log)
    if a.matcher == "seq":
        # sequential + loop closure: video frames are ordered, and exhaustive
        # matching on hundreds of images is minutes of pointless work
        sh(["colmap", "sequential_matcher", "--database_path", db,
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "0",
            f"--{mopt}.use_gpu", gpu], log)
    else:
        sh(["colmap", "exhaustive_matcher", "--database_path", db,
            f"--{mopt}.use_gpu", gpu], log)

    # ---- now the ids exist: write the poses against THEM ---------------------
    # Also take the camera_id from the database rather than assuming 1, for the
    # same reason -- it is COLMAP's to assign, not ours.
    import sqlite3
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    cam_rows = con.execute(
        "SELECT camera_id, model, width, height, params FROM cameras").fetchall()
    con.close()

    lines, used = [], 0
    for image_id, name, camera_id in sorted(rows):
        if name not in pose_by_name:
            continue                      # indexed but outside the trajectory
        used += 1
        lines.append(f"{image_id} {pose_by_name[name]} {camera_id} {name}\n\n")
    if used < 20:
        raise SystemExit(f"only {used} of {len(rows)} database images had a "
                         f"pose -- names in the database do not line up with "
                         f"the frames we posed")
    (pose_dir / "images.txt").write_text("".join(lines))
    # rewrite cameras.txt with the database's own camera row, so the ids agree
    if cam_rows:
        cid, model_id, w, h, blob = cam_rows[0]
        params = np.frombuffer(blob, dtype=np.float64)
        model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                      3: "RADIAL", 4: "OPENCV"}.get(model_id, "OPENCV")
        (pose_dir / "cameras.txt").write_text(
            f"{cid} {model_name} {w} {h} "
            + " ".join(f"{v:.12g}" for v in params) + "\n")
    print(f"  posed {used} of {len(rows)} database images")

    # ---- triangulate into the fixed poses -----------------------------------
    out.mkdir(parents=True, exist_ok=True)
    sh(["colmap", "point_triangulator", "--database_path", db,
        "--image_path", frames, "--input_path", pose_dir,
        "--output_path", out,
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0"], log)

    # COLMAP writes binary unless asked; normalise to text for our tools
    if (out / "points3D.bin").exists() and not (out / "points3D.txt").exists():
        sh(["colmap", "model_converter", "--input_path", out,
            "--output_path", out, "--output_type", "TXT"], log)

    npts = sum(1 for l in open(out / "points3D.txt")
               if l.strip() and not l.startswith("#"))
    nimg = sum(1 for l in open(out / "images.txt")
               if l.strip() and not l.startswith("#")) // 2
    print(f"triangulated {npts} points over {nimg} images (poses held fixed) "
          f"-> {out}")
    if npts < 200:
        raise SystemExit(f"only {npts} points triangulated -- too sparse for "
                         f"the refinement BA to have anything to hold on to")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
