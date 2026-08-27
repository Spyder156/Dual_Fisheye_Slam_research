#!/usr/bin/env python3
"""Build a COLMAP model from a trajectory whose poses we already trust.

KB variant: fisheye (KB4 / OPENCV_FISHEYE) and our dataset layout, instead of
the Mecka pipeline's OPENCV pinhole + episode/offset files.

Why this exists (unchanged from the original): the refinement BA needs a model
-- poses AND points AND the observations linking them. A VIO/SLAM front-end
gives excellent poses and no structure. So: hold the poses FIXED, detect and
match features, and triangulate structure into them. The poses do not move here;
moving them is the BA's job in the next stage, where the IMU gets a vote.

Frame conventions, the easy thing to get wrong:
  input TUM   world-from-BODY (IMU)
  COLMAP      cam-from-world
so each pose goes through the camera<-IMU extrinsic before being written.

Usage:
  triangulate_from_poses.py --traj f_orb.txt --frames-dir <ds>/cam0 \
      --frames-csv <ds>/frames.csv --imucam kalibr_imucam_chain.yaml \
      --out <model_dir> [--matcher seq|exh] [--max-frames 400]
"""
import argparse
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sh(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} "
                           f"failed (exit {r.returncode}); see {log}")


def colmap_opt_prefixes(colmap):
    """('FeatureExtraction','FeatureMatching') on COLMAP 4.x, Sift* on 3.x.

    An unrecognised option is NOT a hard failure -- colmap logs it and proceeds
    with the default -- so guessing wrong silently ignores use_gpu and shows up
    much later as an empty model. Probe, never assume.
    """
    try:
        h = subprocess.run([colmap, "feature_extractor", "--help"],
                           capture_output=True, text=True, timeout=60)
        out = (h.stdout or "") + (h.stderr or "")
        if "FeatureExtraction.max_image_size" in out:
            return "FeatureExtraction", "FeatureMatching"
    except (OSError, subprocess.SubprocessError):
        pass
    return "SiftExtraction", "SiftMatching"


def load_imucam(path, cam=0):
    """kalibr_imucam_chain.yaml -> (R_BC, T_BC, fx,fy,cx,cy, dist, w, h).
    T_imu_cam is [R_CtoI | p_CinI], i.e. camera -> IMU(body)."""
    txt = Path(path).read_text()
    blk = re.split(r"\ncam\d+:", txt)[1 + cam]
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                         blk.split("cam_overlaps")[0])]
    T = np.array(nums).reshape(4, 4)
    fx, fy, cx, cy = [float(x) for x in
                      re.search(r"intrinsics:\s*\[(.*?)\]", blk).group(1).split(",")]
    dist = [float(x) for x in
            re.search(r"distortion_coeffs:\s*\[(.*?)\]", blk).group(1).split(",")]
    w, h = [int(float(x)) for x in
            re.search(r"resolution:\s*\[(.*?)\]", blk).group(1).split(",")]
    return T[:3, :3], T[:3, 3], fx, fy, cx, cy, dist, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="TUM: t tx ty tz qx qy qz qw")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--frames-csv", required=True, help="our frames.csv: frame,t")
    ap.add_argument("--imucam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--matcher", default="seq", choices=["seq", "exh"])
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--colmap", default="colmap")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--max-dt", type=float, default=0.02,
                    help="a pose further than this from the frame time is an "
                         "extrapolation, not a pose")
    a = ap.parse_args()

    frames = Path(a.frames_dir)
    out = Path(a.out)
    work = Path(a.work) if a.work else out.parent / (out.name + "_work")
    work.mkdir(parents=True, exist_ok=True)
    log = work / "colmap.log"

    R_CtoI, p_CinI, fx, fy, cx, cy, dist, W, H = load_imucam(a.imucam)
    R_BC, T_BC = R_CtoI, p_CinI          # body <- camera
    print(f"calib: fx={fx:.2f} cx={cx:.2f} {W}x{H}  fisheye k={np.round(dist,5)}")

    # ---- frame index -> time, from OUR frames.csv ---------------------------
    fr = np.genfromtxt(a.frames_csv, delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int)
    ft = np.atleast_1d(fr["t"])
    tmap = dict(zip(fid, ft))

    traj = np.loadtxt(a.traj, comments="#")
    t_tr = traj[:, 0]
    if t_tr[0] > 1e12:                    # ORB-SLAM3 writes ns
        t_tr = t_tr / 1e9

    imgs = sorted(p for p in frames.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        raise SystemExit(f"no frames in {frames}")
    if len(imgs) > a.max_frames:
        keep = np.linspace(0, len(imgs) - 1, a.max_frames).astype(int)
        imgs = [imgs[i] for i in keep]

    # Stage ONLY the selected frames. Pointing colmap at the full folder makes
    # it index every frame, so sequential matching pairs mostly UNPOSED images
    # and point_triangulator gets nothing (observed: 383 posed of 3881 indexed,
    # 0 points triangulated).
    stage = work / "frames"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p_ in imgs:
        os.symlink(p_.resolve(), stage / p_.name)
    frames = stage
    print(f"staged {len(imgs)} frames for colmap")

    # ---- a model with known poses and no points -----------------------------
    pose_dir = work / "known_poses"
    pose_dir.mkdir(exist_ok=True)
    (pose_dir / "points3D.txt").write_text("")
    (pose_dir / "cameras.txt").write_text(
        f"1 OPENCV_FISHEYE {W} {H} {fx} {fy} {cx} {cy} "
        + " ".join(str(d) for d in dist[:4]) + "\n")

    pose_by_name = {}
    for p in imgs:
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        if idx not in tmap:
            continue
        k = int(np.abs(t_tr - tmap[idx]).argmin())
        if abs(t_tr[k] - tmap[idx]) > a.max_dt:
            continue
        p_wb = traj[k, 1:4]
        R_wb = Rotation.from_quat(traj[k, 4:8]).as_matrix()
        R_wc = R_wb @ R_BC                       # body -> camera
        p_wc = p_wb + R_wb @ T_BC
        R_cw = R_wc.T
        t_cw = -R_cw @ p_wc
        q = Rotation.from_matrix(R_cw).as_quat()      # xyzw
        pose_by_name[p.name] = (f"{q[3]} {q[0]} {q[1]} {q[2]} "
                                f"{t_cw[0]} {t_cw[1]} {t_cw[2]}")
    if len(pose_by_name) < 20:
        raise SystemExit(f"only {len(pose_by_name)} frames had a pose within "
                         f"{a.max_dt*1000:.0f} ms -- trajectory and frame clock disagree")
    print(f"posed {len(pose_by_name)} of {len(imgs)} selected frames")

    # ---- features and matches ----------------------------------------------
    db = work / "database.db"
    if db.exists():
        db.unlink()
    xopt, mopt = colmap_opt_prefixes(a.colmap)
    sh([a.colmap, "feature_extractor", "--database_path", db,
        "--image_path", frames,
        "--ImageReader.camera_model", "OPENCV_FISHEYE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params",
        f"{fx},{fy},{cx},{cy}," + ",".join(str(d) for d in dist[:4]),
        f"--{xopt}.use_gpu", a.gpu], log)
    if a.matcher == "seq":
        sh([a.colmap, "sequential_matcher", "--database_path", db,
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "0",
            f"--{mopt}.use_gpu", a.gpu], log)
    else:
        sh([a.colmap, "exhaustive_matcher", "--database_path", db,
            f"--{mopt}.use_gpu", a.gpu], log)

    # ---- now the ids exist: write the poses against THEM ---------------------
    # COLMAP assigns image_id when the database indexes the files;
    # point_triangulator matches by ID not by name, so writing our own
    # sequential ids gives "Check failed: existing_image.Name() == ...".
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    cam_rows = con.execute(
        "SELECT camera_id, model, width, height, params FROM cameras").fetchall()
    con.close()

    lines, used = [], 0
    for image_id, name, camera_id in sorted(rows):
        if name not in pose_by_name:
            continue
        used += 1
        lines.append(f"{image_id} {pose_by_name[name]} {camera_id} {name}\n\n")
    if used < 20:
        raise SystemExit(f"only {used} of {len(rows)} database images had a pose")
    (pose_dir / "images.txt").write_text("".join(lines))
    if cam_rows:
        cid, model_id, w, h, blob = cam_rows[0]
        params = np.frombuffer(blob, dtype=np.float64)
        model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                      3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                      6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
                      9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}.get(model_id)
        if model_name is None:
            raise SystemExit(f"unknown COLMAP camera model id {model_id}")
        (pose_dir / "cameras.txt").write_text(
            f"{cid} {model_name} {w} {h} "
            + " ".join(f"{v:.12g}" for v in params) + "\n")
    print(f"  posed {used} of {len(rows)} database images")

    # ---- triangulate into the fixed poses -----------------------------------
    out.mkdir(parents=True, exist_ok=True)
    sh([a.colmap, "point_triangulator", "--database_path", db,
        "--image_path", frames, "--input_path", pose_dir,
        "--output_path", out,
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0"], log)

    if (out / "points3D.bin").exists() and not (out / "points3D.txt").exists():
        sh([a.colmap, "model_converter", "--input_path", out,
            "--output_path", out, "--output_type", "TXT"], log)

    npts = sum(1 for l in open(out / "points3D.txt")
               if l.strip() and not l.startswith("#"))
    nimg = sum(1 for l in open(out / "images.txt")
               if l.strip() and not l.startswith("#")) // 2
    print(f"triangulated {npts} points over {nimg} images (poses fixed) -> {out}")
    if npts < 200:
        raise SystemExit(f"only {npts} points triangulated -- too sparse for the "
                         f"refinement BA to hold on to")


if __name__ == "__main__":
    main()
