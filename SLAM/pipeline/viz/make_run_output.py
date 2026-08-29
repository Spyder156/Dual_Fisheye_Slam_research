#!/usr/bin/env python
"""Produce the standard run output (see SLAM/docs/OUTPUT.md) for a SLAM run.

Builds run.rrd with:
  - big 3D pane: camera path + SLAM point cloud (+ GT path if given)
  - side pane: video with current keypoints bright and tracked history fading
  - two plots: reprojection error, gyro-vs-estimate rotation error
plus viz/ stills.

Usage:
  make_run_output.py --engine openvins --traj T.csv --dataset D --out RUNDIR [--gt gt.txt]
  make_run_output.py --engine okvis    --okvis-dir D --dataset D2 --out RUNDIR [--gt gt.txt]
"""
import argparse
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")

TRAIL = 12          # how many past observations of a track to keep drawing
MAX_POINTS = 120000


def _nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def load_cameras_orb(yml):
    """Cameras from the ORB-SLAM3 settings file the run actually used.

    Reading the estimator's OWN config is the only way to guarantee the frusta
    and the display rotation match what SLAM saw. Pointing this at some other
    calibration directory is how the frames silently came out upside down and
    the camera cones vanished: load_cameras() returned [], and every consumer
    treated "no cameras" as "nothing to draw" rather than as an error.
    """
    txt = Path(yml).read_text()

    def g(k, d=0.0):
        m = re.search(rf"^{re.escape(k)}:\s*([-\d.eE+]+)", txt, re.M)
        return float(m.group(1)) if m else d

    def mat(key):
        m = re.search(rf"{re.escape(key)}:.*?data:\s*\[(.*?)\]", txt, re.S)
        return np.array(_nums(m.group(1))).reshape(4, 4) if m else None

    w, h = int(g("Camera.width")), int(g("Camera.height"))
    T_b_c1 = mat("IMU.T_b_c1")
    if T_b_c1 is None:
        return []
    cams = [dict(T=T_b_c1, fx=g("Camera1.fx"), fy=g("Camera1.fy"),
                 cx=g("Camera1.cx"), cy=g("Camera1.cy"), w=w, h=h,
                 d=[g("Camera1.k1"), g("Camera1.k2"), g("Camera1.k3"), g("Camera1.k4")],
                 mask=None)]
    if g("Camera2.fx"):
        # rear lens: T_b_c2 = T_b_c1 * T_c1_c2, the rig transform
        T_c1_c2 = mat("Rig.T_c0_c1")
        if T_c1_c2 is None:
            T_c1_c2 = mat("Stereo.T_c1_c2")
        if T_c1_c2 is not None:
            cams.append(dict(T=T_b_c1 @ T_c1_c2, fx=g("Camera2.fx"), fy=g("Camera2.fy"),
                             cx=g("Camera2.cx"), cy=g("Camera2.cy"), w=w, h=h,
                             d=[g("Camera2.k1"), g("Camera2.k2"),
                                g("Camera2.k3"), g("Camera2.k4")], mask=None))
    return cams


def load_cameras(cfg):
    """Read camera intrinsics + T_CtoI (camera->IMU) from either config style.

    OpenVINS: kalibr_imucam_chain.yaml (T_imu_cam as rows). OKVIS2: hilti.yaml
    (T_SC as a flat 16-list). Both are OpenCV-flavoured YAML, so parse the
    numbers directly rather than fighting the %YAML:1.0 directive.
    """
    cfg = Path(cfg)
    if cfg.is_file():
        return load_cameras_orb(cfg)
    cams = []
    kal = cfg / "kalibr_imucam_chain.yaml"
    ok = cfg / "hilti.yaml"
    if kal.exists():
        txt = kal.read_text()
        for blk in re.split(r"\ncam\d+:", txt)[1:]:
            T = np.array(_nums(blk.split("cam_overlaps")[0])).reshape(4, 4)
            fx, fy, cx, cy = _nums(re.search(r"intrinsics:.*", blk).group(0))
            w, h = _nums(re.search(r"resolution:.*", blk).group(0))
            d = _nums(re.search(r"distortion_coeffs:.*", blk).group(0))
            cams.append(dict(T=T, fx=fx, fy=fy, cx=cx, cy=cy, w=int(w), h=int(h), d=d))
    elif ok.exists():
        txt = ok.read_text()
        for m in re.finditer(r"T_SC:\s*\[(.*?)\]", txt, re.S):
            T = np.array(_nums(m.group(1))).reshape(4, 4)
            tail = txt[m.end():m.end() + 800]
            fx, fy = _nums(re.search(r"focal_length:\s*\[(.*?)\]", tail).group(1))
            cx, cy = _nums(re.search(r"principal_point:\s*\[(.*?)\]", tail).group(1))
            w, h = _nums(re.search(r"image_dimension:\s*\[(.*?)\]", tail).group(1))
            d = _nums(re.search(r"distortion_coefficients:\s*\[(.*?)\]", tail).group(1))
            cams.append(dict(T=T, fx=fx, fy=fy, cx=cx, cy=cy, w=int(w), h=int(h), d=d))
    for c, cam in enumerate(cams):
        mp = cfg / f"mask{c}.png"
        cam["mask"] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
    return cams


def shade_masked(gray, mask):
    """Show the masked-out region rather than hiding it: darken + red tint, so
    what the tracker is forbidden to use stays visible as context."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if mask is None:
        return rgb
    if mask.shape != gray.shape:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = mask > 127                       # OpenVINS convention: nonzero = ignore
    rgb[m] = np.clip(rgb[m] * 0.38 + np.array([64, 16, 16]), 0, 255).astype(np.uint8)
    return rgb


def project_kb4(Pc, cam):
    """Equidistant (KB4) projection — the model these cameras are calibrated in."""
    x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    k1, k2, k3, k4 = cam["d"]
    td = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    s = np.where(r > 1e-9, td / np.maximum(r, 1e-9), 0.0)
    return cam["fx"] * x * s + cam["cx"], cam["fy"] * y * s + cam["cy"], th


def scene_colors(pts, P, Rw, T, cams, ds, fids, fts, every=15, max_theta=1.60):
    """Colour each landmark with the ACTUAL scene intensity it came from.

    For every landmark, find nearby camera poses, project it with the real KB4
    model, and sample the pixel. Note the Hilti cameras are monochrome
    (`camera_type: gray`), so "scene colour" is a grey level, not RGB — this
    shades the map by real appearance, it does not invent hue.
    """
    from scipy.spatial import cKDTree
    ks = list(range(0, len(fids), every))
    kt = np.array([fts[k] for k in ks])
    idx = np.clip(np.searchsorted(T, kt), 0, len(T) - 1)      # pose per sampled frame
    cen = P[idx]
    tree = cKDTree(cen)
    col = np.full(len(pts), -1.0)
    _, order = tree.query(pts, k=min(6, len(ks)))
    order = np.atleast_2d(order)
    cache = {}
    for slot in range(order.shape[1]):
        todo = np.where(col < 0)[0]
        if not len(todo):
            break
        for f in np.unique(order[todo, slot]):
            sel = todo[order[todo, slot] == f]
            k, i = ks[f], idx[f]
            for c, cam in enumerate(cams):
                R_CtoG = Rw[i] @ cam["T"][:3, :3]
                p_CinG = P[i] + Rw[i] @ cam["T"][:3, 3]
                Pc = (pts[sel] - p_CinG) @ R_CtoG                  # world -> camera
                u, v, th = project_kb4(Pc, cam)
                ok = (th < max_theta) & (u >= 0) & (v >= 0) & \
                     (u < cam["w"] - 1) & (v < cam["h"] - 1)
                if not ok.any():
                    continue
                key = (c, k)
                if key not in cache:
                    if len(cache) > 40:
                        cache.pop(next(iter(cache)))
                    cache[key] = cv2.imread(
                        str(ds / f"cam{c}" / f"{fids[k]:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
                img = cache[key]
                if img is None:
                    continue
                sy = img.shape[0] / cam["h"]
                got = sel[ok]
                col[got] = img[np.clip((v[ok] * sy).astype(int), 0, img.shape[0] - 1),
                               np.clip((u[ok] * sy).astype(int), 0, img.shape[1] - 1)]
    n_col = int((col >= 0).sum())
    g = np.where(col >= 0, col, 110).astype(np.uint8)
    return np.stack([g, g, g], 1), n_col


def clip_cloud(pts, P, margin=1.5, floor_m=8.0):
    """Drop landmarks far from the trajectory.

    Rerun auto-fits the 3D view to ALL data, so a handful of near-infinity
    landmarks (OKVIS2 keeps plenty: its map reaches 4000 km) shrink a 30 m
    trajectory to a dot. Keep points within `margin * trajectory diagonal` of
    the nearest pose. Returns (kept, n_dropped) — the count is printed, never
    silently swallowed.
    """
    if not len(pts):
        return pts, 0
    from scipy.spatial import cKDTree
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    r = max(floor_m, margin * diag)
    d, _ = cKDTree(P).query(pts, k=1)
    keep = d <= r
    return pts[keep], int((~keep).sum())


def set_t(t):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("t", t)
    else:
        rr.set_time("t", duration=t)


def load_csv_loose(path, ncol):
    rows = []
    for i, line in enumerate(open(path)):
        if i == 0:
            continue
        v = [x for x in line.strip().rstrip(",").split(",") if x.strip() != ""]
        if len(v) >= ncol:
            try:
                rows.append([float(x) for x in v[:ncol]])
            except ValueError:
                pass
    return np.array(rows)


def umeyama(A, B):
    mA, mB = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - mA).T @ (B - mB))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mB - R @ mA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["openvins", "okvis", "tum"], required=True)
    ap.add_argument("--traj", type=Path)
    ap.add_argument("--okvis-dir", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1, help="video frame stride (1 = full fps)")
    ap.add_argument("--elsed", default=None,
                    help="line_extract binary; enables the two LINE panes")
    ap.add_argument("--line-masks", default=None,
                    help="dir with selfocc_cam{0,1}.png, applied as in the SLAM")
    ap.add_argument("--max-depth", type=float, default=30.0,
                    help="drop landmarks further than this from the camera that "
                         "created them (display only)")
    ap.add_argument("--clip-margin", type=float, default=1.5,
                    help="keep landmarks within margin*traj_diagonal of the path")
    args = ap.parse_args()

    out = args.out
    (out / "viz").mkdir(parents=True, exist_ok=True)

    # ---------------- load trajectory + points + per-frame keypoints ----------
    kps = None          # dict t -> list of (id, u, v)
    stats = None
    if args.engine == "openvins":
        d = load_csv_loose(args.traj, 8)
        T, P, Q = d[:, 0], d[:, 1:4], d[:, 4:8]
        pf = Path(str(args.traj) + ".points.csv")
        pts = np.unique(load_csv_loose(pf, 5)[:, 2:5], axis=0) if pf.exists() else np.zeros((0, 3))
        kf = Path(str(args.traj) + ".keypoints.csv")
        if kf.exists():
            raw = load_csv_loose(kf, 5)
            kps = defaultdict(list)
            for r in raw:
                kps[round(r[0], 6)].append((int(r[1]), r[2], r[3]))
        sf = Path(str(args.traj) + ".stats.csv")
        stats = load_csv_loose(sf, 3) if sf.exists() else None
    elif args.engine == "tum":
        # ORB-SLAM3 / TUM format: t[ns or s] tx ty tz qx qy qz qw, no header.
        d = np.loadtxt(args.traj)
        T = d[:, 0] / (1e9 if d[0, 0] > 1e12 else 1.0)
        P, Q = d[:, 1:4], d[:, 4:8]
        # sparse map, if SaveMapPoints() was run. Column 0 is the timestamp of
        # the first observing keyframe, so the cloud can GROW over time.
        mf = Path(str(args.traj).replace("f_", "mp_").replace(".txt", ".csv"))
        if mf.exists():
            m = load_csv_loose(mf, 5)
            if m.shape[1] >= 5:
                pt_t, pts, pt_cam = m[:, 0], m[:, 1:4], m[:, 4].astype(int)
            else:
                m = load_csv_loose(mf, 4)
                pt_t, pts, pt_cam = m[:, 0], m[:, 1:4], None
        else:
            pt_t, pts, pt_cam = None, np.zeros((0, 3)), None
        # per-frame keypoints, columns t,cam,id,u,v,tracked
        kf2 = Path(str(args.traj).replace("f_", "kp_").replace(".txt", ".csv"))
        if kf2.exists():
            raw = load_csv_loose(kf2, 6)
            kps = defaultdict(list)
            for r in raw:
                kps[round(r[0], 6)].append((int(r[1]), int(r[2]), r[3], r[4], int(r[5])))
            print(f"loaded {len(raw)} keypoint rows over {len(kps)} frames")
    else:
        traj = args.okvis_dir / "okvis2-slam-final_trajectory.csv"
        d = load_csv_loose(traj, 8)
        T, P, Q = d[:, 0] / 1e9, d[:, 1:4], d[:, 4:8]
        mp = args.okvis_dir / "okvis2-slam-final_map.csv"
        pts = load_csv_loose(mp, 4)[:, 1:4] if mp.exists() else np.zeros((0, 3))

    pt_t = pt_t if args.engine == "tum" else None
    pt_cam = pt_cam if args.engine == "tum" else None
    # points.csv keeps the FULL map; clipping below is for display only
    np.savetxt(out / "points.csv", pts, delimiter=",", header="x,y,z", comments="")

    pts_all_t = pts
    pt_t_kept = pt_cam_kept = None
    if pt_t is not None:
        # CULL BY DEPTH FROM THE CAMERA THAT CREATED THE LANDMARK.
        #
        # The old test -- distance to the nearest point of the path -- scales
        # its radius with the trajectory's own diagonal. When a run drifts, the
        # trajectory grows, the radius grows with it, and the filter opens up
        # exactly when the map is at its worst. On run_2 that left a 310 m
        # radius and passed 99.6% of a map whose landmarks reach 300 m out.
        #
        # Depth from the CREATING camera does not have that feedback: indoors, a
        # landmark hundreds of metres from the camera that first saw it is a
        # failed triangulation, however far the estimate has wandered. This is a
        # DISPLAY cull only -- points.csv above keeps the full map.
        A = np.stack([np.interp(pt_t, T, P[:, i]) for i in range(3)], 1)
        depth = np.linalg.norm(pts - A, axis=1)
        keep = depth <= args.max_depth
        n_far = int((~keep).sum())
        pts, pt_t_kept = pts[keep], pt_t[keep]
        if pt_cam is not None:
            pt_cam_kept = pt_cam[keep]
        if n_far:
            print(f"culled {n_far} landmarks beyond {args.max_depth:.0f} m of the "
                  f"camera that created them ({100*n_far/len(keep):.1f}% of the map; "
                  f"median depth of the rest {np.median(depth[keep]):.1f} m)")
    else:
        pts, n_far = clip_cloud(pts, P, margin=args.clip_margin)
    if n_far and pt_t is None:
        print(f"clipped {n_far} landmarks far from the trajectory "
              f"({100*n_far/(n_far+len(pts)):.1f}% of the map)")
    if len(pts) > MAX_POINTS:
        _sub = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[_sub]
        if pt_t_kept is not None:
            pt_t_kept = pt_t_kept[_sub]
        if pt_cam_kept is not None:
            pt_cam_kept = pt_cam_kept[_sub]

    # ---------------- put the ESTIMATE into the GT frame ---------------------
    # Align the ESTIMATE onto GT, not GT onto the estimate. Both conventions
    # give the same error number, but only this one leaves GT sitting in its
    # true frame -- so when the run drifts you SEE it peel away from the real
    # path. The map has to ride the same transform as the trajectory that
    # created it; transforming one without the other is what made the cloud
    # look "aligned to the old trajectory".
    align = None
    if args.gt and args.gt.exists():
        _g = np.loadtxt(args.gt)
        _tg, _Pg = _g[:, 0], _g[:, 1:4]
        _m = (_tg >= T[0]) & (_tg <= T[-1])
        if _m.sum() > 10:
            _A = np.stack([np.interp(_tg[_m], T, P[:, i]) for i in range(3)], 1)
            align = umeyama(_A, _Pg[_m])          # SLAM -> GT
            R_al, t_al = align
            P = (R_al @ P.T).T + t_al
            Q = Rotation.from_matrix(R_al @ Rotation.from_quat(Q).as_matrix()).as_quat()
            if len(pts):
                pts = (R_al @ pts.T).T + t_al
            if len(pts_all_t):
                pts_all_t = (R_al @ pts_all_t.T).T + t_al
            print("estimate aligned into the GT frame (map rides the same transform)")

    # every visual size derives from scene scale, so the view reads the same
    # whether the run is a 3 m desk loop or a 300 m building
    S = float(np.linalg.norm(P.max(0) - P.min(0)))
    ctr = 0.5 * (P.max(0) + P.min(0))

    np.savetxt(out / "traj.csv", np.column_stack([T, P, Q]), delimiter=",",
               header="t,px,py,pz,qx,qy,qz,qw", comments="")
    if args.config and args.config.exists():
        if args.config.is_file():
            (out / "config").mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.config, out / "config" / args.config.name)
        else:
            shutil.copytree(args.config, out / "config", dirs_exist_ok=True)

    # ---------------- GT, aligned into the SLAM frame ------------------------
    Pg = tg = None
    if args.gt and args.gt.exists():
        g = np.loadtxt(args.gt)
        tg_all, Pg_all = g[:, 0], g[:, 1:4]
        m = (tg_all >= T[0]) & (tg_all <= T[-1])
        if m.sum() > 10:
            A = np.stack([np.interp(tg_all[m], T, P[:, i]) for i in range(3)], 1)
            # P is already in the GT frame, so GT is drawn untouched
            Pg, tg = Pg_all[m], tg_all[m]
            err = np.linalg.norm(A - Pg, axis=1)
            score = float((100 * np.exp(-0.46051701859880917 * err)).mean())
            (out / "score.txt").write_text(
                f"ATE RMSE : {np.sqrt((err**2).mean()):.4f} m\n"
                f"coverage : {100*m.sum()/len(tg_all):.2f} %\n"
                f"SCORE    : {score:.2f}\n")

    # ---------------- rerun ---------------------------------------------------
    # Place the eye explicitly: auto-fit is unreliable once any outlier survives,
    # and a 3/4 view from above shows a walked floor plan best. 1 m grid gives scale.
    eye = ctr + np.array([0.55, -0.95, 0.75]) * max(S, 1.0)
    bp = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(
            origin="/world", name="map + trajectory", contents="/world/**",
            eye_controls=rrb.archetypes.EyeControls3D(
                position=eye.tolist(), look_target=ctr.tolist(), eye_up=[0, 0, 1]),
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True, spacing=1.0, stroke_width=1.0, color=[70, 70, 78, 140])),
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/cam0", name="FRONT points (blue)",
                                  contents=["/cam0/image", "/cam0/image/keypoints"]),
                rrb.Spatial2DView(origin="/cam1", name="REAR points (red)",
                                  contents=["/cam1/image", "/cam1/image/keypoints"])),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/lines0", name="FRONT lines (blue)",
                                  contents="/lines0/**"),
                rrb.Spatial2DView(origin="/lines1", name="REAR lines (red)",
                                  contents="/lines1/**")),
            rrb.TimeSeriesView(origin="/plots/kp_count", name="tracked keypoints per camera"),
            row_shares=[3, 3, 1]),
        column_shares=[3, 2]))
    rr.init("slam_run", spawn=False)
    rr.save(str(out / "run.rrd"))
    rr.send_blueprint(bp, make_active=True, make_default=True)

    # point cloud: log once at the first timestamp (not static — that hangs 0.33)
    set_t(float(T[0]))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    cams = load_cameras(args.config) if args.config and args.config.exists() else []
    Rw = Rotation.from_quat(Q).as_matrix()      # R_ItoG

    frames = np.genfromtxt(args.dataset / "frames.csv", delimiter=",", names=True)
    fids = np.atleast_1d(frames["frame"]).astype(int)
    fts = np.atleast_1d(frames["t"])

    # ---- colour the map ------------------------------------------------------
    # Pick the colours first, then log ONCE through the shared growth path
    # below. Keeping these separate is deliberate: the previous version computed
    # rig colours in a branch that never reached a rr.log() call, so every rig
    # run printed correct red/blue/green counts and shipped an empty cloud.
    cols = None
    if len(pts) and pt_cam_kept is not None:
        # RIG CHECK: colour by the camera that saw the landmark, not by scene
        # appearance. If the rig geometry is right, BLUE (front) sits ahead of
        # the camera and RED (rear) behind it. That is directly visible.
        cols = np.zeros((len(pts), 3), np.uint8)
        cols[pt_cam_kept == 0] = [60, 130, 255]     # front  -> blue
        cols[pt_cam_kept == 1] = [255, 70, 70]      # rear   -> red
        cols[pt_cam_kept == 2] = [40, 210, 90]      # both   -> GREEN (cross-lens match)
        n0 = int((pt_cam_kept == 0).sum()); n1 = int((pt_cam_kept == 1).sum())
        nb = int((pt_cam_kept == 2).sum())
        print(f"cloud by camera: front(blue) {n0}, rear(red) {n1}, both(GREEN) {nb}")
    elif len(pts) and cams:
        cols, n_col = scene_colors(pts, P, Rw, T, cams, args.dataset, fids, fts)
        print(f"scene-coloured {n_col}/{len(pts)} landmarks "
              f"({100*n_col/len(pts):.1f}%); rest left neutral grey")
    elif len(pts):
        cols = np.tile(np.array([[150, 150, 160]], np.uint8), (len(pts), 1))

    if cols is not None:
        if pt_t_kept is not None and len(pt_t_kept) == len(pts):
            # GROW the map: at each time bucket log every landmark created so
            # far, so scrubbing the timeline shows the map being built rather
            # than the finished cloud sitting there from t=0.
            tb = np.linspace(T[0], T[-1], 240)
            order = np.argsort(pt_t_kept)
            ts_sorted = pt_t_kept[order]
            for tt in tb:
                k = int(np.searchsorted(ts_sorted, tt))
                if k < 8:
                    continue
                set_t(float(tt))
                sel = order[:k]
                rr.log("world/points",
                       rr.Points3D(pts[sel], colors=cols[sel], radii=S * 0.00025))
            print(f"map grows over {len(tb)} steps -> {len(pts)} landmarks")
        else:
            print("no per-landmark timestamps; logging the map statically")
            rr.log("world/points", rr.Points3D(pts, colors=cols, radii=S * 0.00025))

    # real camera frusta at the live pose — sized like a real camera (~15 cm at
    # room scale), not a landmark of the map
    for c, cam in enumerate(cams):
        rr.log(f"world/cam{c}", rr.Pinhole(
            focal_length=[cam["fx"], cam["fy"]],
            principal_point=[cam["cx"], cam["cy"]],
            resolution=[cam["w"], cam["h"]],
            image_plane_distance=S * 0.005))

    # trajectory grows over time
    # ~1/3 of the old width. Positive radii are WORLD units, so the line
    # thickens and thins with zoom instead of staying a fixed screen smear.
    w_line = S * 0.0008
    for i in range(1, len(P)):
        set_t(float(T[i]))
        rr.log("world/traj", rr.LineStrips3D([P[:i + 1]], colors=[[42, 120, 214]],
                                             radii=w_line))
        for c, cam in enumerate(cams):
            R_CtoI, p_CinI = cam["T"][:3, :3], cam["T"][:3, 3]
            rr.log(f"world/cam{c}", rr.Transform3D(
                translation=P[i] + Rw[i] @ p_CinI, mat3x3=Rw[i] @ R_CtoI))
    if Pg is not None:
        for i in range(1, len(Pg)):
            set_t(float(tg[i]))
            rr.log("world/gt", rr.LineStrips3D([Pg[:i + 1]], colors=[[27, 175, 122]],
                                               radii=w_line))

    # ---------------- video with fading track tails --------------------------
    # The Hilti cameras are mounted rotated ~180 deg: world-up projects onto the
    # image +y (down) axis, so the raw frames read upside down to a human. The
    # calibration encodes this, so SLAM is correct on the raw frames -- we rotate
    # for DISPLAY only, and rotate the keypoint coordinates to match.
    rot180 = False
    if cams:
        imu_a = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
        up_I = np.stack([imu_a["ax"], imu_a["ay"], imu_a["az"]], 1).mean(0)
        up_C = cams[0]["T"][:3, :3].T @ (up_I / np.linalg.norm(up_I))
        rot180 = up_C[1] > 0.5            # world-up points DOWN in the image
        print(f"display rotation: {'180 deg (sensor frame is upside down)' if rot180 else 'none'}")
    # ---- lines, same ELSED the estimator runs -------------------------------
    lines = {0: {}, 1: {}}
    lmask = {0: None, 1: None}
    if args.elsed:
        import subprocess
        if args.line_masks:
            for c in (0, 1):
                mp = Path(args.line_masks) / f"selfocc_cam{c}.png"
                if mp.exists():
                    lmask[c] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        for c in (0, 1):
            lst = out / f"_lines_cam{c}.list"
            csvp = out / f"_lines_cam{c}.csv"
            lst.write_text("\n".join(str(i) for i in fids) + "\n")
            r = subprocess.run([args.elsed, str(args.dataset / f"cam{c}"),
                                str(lst), str(csvp), "30", "15"],
                               capture_output=True, text=True)
            if r.returncode:
                print(f"line_extract cam{c} failed: {r.stderr[-200:]}")
                continue
            with open(csvp) as f:
                f.readline()
                for l in f:
                    p_ = l.split(",")
                    lines[c].setdefault(int(p_[0]), []).append(
                        [float(p_[1]), float(p_[2]), float(p_[3]), float(p_[4])])
            print(f"lines cam{c}: {sum(len(v) for v in lines[c].values())} segments")

    hist = defaultdict(lambda: deque(maxlen=TRAIL))
    kts = np.array(sorted(kps.keys())) if kps else None

    for k in range(0, len(fids), args.stride):
        t = float(fts[k])
        img = cv2.imread(str(args.dataset / "cam0" / f"{fids[k]:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        set_t(t)
        m0 = cams[0]["mask"] if cams else None
        vis = shade_masked(img, m0)
        H, W = vis.shape[:2]
        if rot180:
            vis = cv2.rotate(vis, cv2.ROTATE_180)
        rr.log("cam0/image", rr.Image(vis).compress(jpeg_quality=70))
        # REAR camera, same treatment. A rig run must show both or we cannot
        # tell which camera lost tracking.
        img1 = cv2.imread(str(args.dataset / "cam1" / f"{fids[k]:06d}.jpg"),
                          cv2.IMREAD_GRAYSCALE)
        if img1 is not None:
            m1 = cams[1]["mask"] if len(cams) > 1 else None
            v1 = shade_masked(img1, m1)
            if rot180:
                v1 = cv2.rotate(v1, cv2.ROTATE_180)
            rr.log("cam1/image", rr.Image(v1).compress(jpeg_quality=70))
        # LINE panes: same red/blue scheme as the points and the 3D cloud.
        # Masked segments are dropped exactly as the estimator drops them, so
        # what is drawn is what the SLAM actually had.
        if args.elsed:
            for c in (0, 1):
                segs = lines[c].get(int(fids[k]), [])
                if not segs:
                    continue
                img_c = img if c == 0 else img1
                if img_c is None:
                    continue
                base = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
                if rot180:
                    base = cv2.rotate(base, cv2.ROTATE_180)
                rr.log(f"lines{c}/image", rr.Image(base).compress(jpeg_quality=70))
                col = [60, 130, 255] if c == 0 else [255, 70, 70]
                strips, cols = [], []
                for (x1, y1, x2, y2) in segs:
                    if lmask[c] is not None:
                        xi1, yi1 = int(x1), int(y1); xi2, yi2 = int(x2), int(y2)
                        h_, w_ = lmask[c].shape
                        if (0 <= yi1 < h_ and 0 <= xi1 < w_ and lmask[c][yi1, xi1]) or \
                           (0 <= yi2 < h_ and 0 <= xi2 < w_ and lmask[c][yi2, xi2]):
                            continue          # on the rig's own hardware
                    if rot180:
                        p1 = [W - 1 - x1, H - 1 - y1]; p2 = [W - 1 - x2, H - 1 - y2]
                    else:
                        p1 = [x1, y1]; p2 = [x2, y2]
                    strips.append([p1, p2]); cols.append(col)
                if strips:
                    rr.log(f"lines{c}/image/segments",
                           rr.LineStrips2D(strips, colors=cols, radii=2.2))
                    rr.log(f"plots/kp_count/lines{c}", rr.Scalars(float(len(strips))))

        if kts is None or not len(kts):
            continue
        j = int(np.argmin(np.abs(kts - t)))
        if abs(kts[j] - t) > 0.05:
            continue
        cur = kps[kts[j]]
        if args.engine == "tum":
            # rows are (cam, id, u, v, tracked). Draw each camera on its own
            # pane in its own colour: FRONT blue, REAR red. Tracked features are
            # bright, untracked dim -- so a tracking collapse is visible as the
            # bright points vanishing, per camera.
            for c in (0, 1):
                sel = [r for r in cur if r[0] == c]
                if not sel:
                    continue
                # Colour is decided by WHICH LENS this observation came from,
                # never by which lens first saw the landmark. A feature drawn on
                # the rear pane is red even if its landmark originated up front.
                base = [60, 130, 255] if c == 0 else [255, 70, 70]
                xy, co = [], []
                for (_, _id, u, v, tr) in sel:
                    xy.append([W - 1 - u, H - 1 - v] if rot180 else [u, v])
                    co.append(base if tr else [int(x * 0.35) for x in base])
                rr.log(f"cam{c}/image/keypoints",
                       rr.Points2D(np.array(xy), colors=co, radii=4.4))
                rr.log(f"plots/kp_count/cam{c}",
                       rr.Scalars(float(sum(1 for r in sel if r[4]))))
            continue
        alive = set()
        for fid, u, v in cur:
            hist[fid].append((u, v))
            alive.add(fid)
        # current observations: bright; history: progressively lighter
        pos_now = np.array([[u, v] for _, u, v in cur], float)
        tail_xy, tail_col = [], []
        for fid in alive:
            h = list(hist[fid])
            n = len(h)
            for a, (u, v) in enumerate(h[:-1]):
                f = (a + 1) / max(n, 2)          # older -> smaller f -> lighter
                shade = int(255 - 110 * f)
                tail_xy.append([u, v])
                tail_col.append([shade, shade, 255])
        def disp(a):
            a = np.asarray(a, float).reshape(-1, 2)
            return np.column_stack([W - 1 - a[:, 0], H - 1 - a[:, 1]]) if rot180 else a

        if tail_xy:
            rr.log("cam0/image/tracks", rr.Points2D(disp(tail_xy), colors=tail_col, radii=2.4))
        if len(pos_now):
            rr.log("cam0/image/keypoints", rr.Points2D(disp(pos_now), colors=[[40, 220, 90]], radii=4.8))
        for fid in list(hist):
            if fid not in alive:
                del hist[fid]

    # ---------------- plots ---------------------------------------------------
    if stats is not None:
        for r in stats:
            if r[1] >= 0:
                set_t(float(r[0]))
                rr.log("plots/reproj_px", rr.Scalars(float(r[1])))

    imu = np.genfromtxt(args.dataset / "imu.csv", delimiter=",", names=True)
    ti, gy = imu["t"], np.stack([imu["gx"], imu["gy"], imu["gz"]], 1)
    Rs = Rotation.from_quat(Q)
    fwd = np.array([1.0, 0, 0])
    yaw_e = np.unwrap([np.arctan2(*(r.as_matrix() @ fwd)[[1, 0]]) for r in Rs])
    Rg = Rs[0].as_matrix().copy()
    yg, tg2 = [np.arctan2(*(Rg @ fwd)[[1, 0]])], [T[0]]
    for k in range(np.searchsorted(ti, T[0]), np.searchsorted(ti, T[-1]) - 1):
        th = gy[k] * (ti[k + 1] - ti[k])
        a = np.linalg.norm(th)
        if a > 1e-12:
            kx = th / a
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            Rg = Rg @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)
        if k % 100 == 0:
            yg.append(np.arctan2(*(Rg @ fwd)[[1, 0]]))
            tg2.append(ti[k])
    yg = np.unwrap(yg)
    yi = np.interp(tg2, T, yaw_e)
    for t_, e in zip(tg2, np.degrees(np.abs((yi - yi[0]) - (yg - yg[0])))):
        set_t(float(t_))
        rr.log("plots/rot_err_deg", rr.Scalars(float(e)))

    # ---------------- stills --------------------------------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(P[:, 0], P[:, 1], color="#2a78d6", lw=1.6, label="SLAM")
    if Pg is not None:
        ax.plot(Pg[:, 0], Pg[:, 1], color="#1baf7a", lw=1.6, label="GT")
    if len(pts):
        s = pts[np.random.default_rng(1).choice(len(pts), min(20000, len(pts)), replace=False)]
        ax.scatter(s[:, 0], s[:, 1], s=0.4, c="#bbbbbb", alpha=0.35, zorder=0)
    ax.set_aspect("equal"); ax.legend(); ax.set_title("camera path, top-down")
    fig.tight_layout(); fig.savefig(out / "viz" / "path_topdown.png", dpi=130); plt.close(fig)

    for k in [int(len(fids) * f) for f in (0.15, 0.45, 0.75)]:
        a = cv2.imread(str(args.dataset / "cam0" / f"{fids[k]:06d}.jpg"))
        b = cv2.imread(str(args.dataset / "cam0" / f"{fids[min(k+5,len(fids)-1)]:06d}.jpg"))
        if a is None or b is None:
            continue
        if rot180:                       # display-only, same as the video pane
            a, b = cv2.rotate(a, cv2.ROTATE_180), cv2.rotate(b, cv2.ROTATE_180)
        g1, g2 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g1, 400, 0.01, 12)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, winSize=(21, 21), maxLevel=4)
        canvas = np.concatenate([a, b], 1)
        w = a.shape[1]
        for (x0, y0), (x1, y1), ok in zip(p0.reshape(-1, 2), p1.reshape(-1, 2), st.ravel()):
            if not ok:
                continue
            cv2.circle(canvas, (int(x0), int(y0)), 3, (80, 220, 80), -1)
            cv2.circle(canvas, (int(x1) + w, int(y1)), 3, (80, 220, 80), -1)
            cv2.line(canvas, (int(x0), int(y0)), (int(x1) + w, int(y1)), (80, 220, 80), 1)
        cv2.imwrite(str(out / "viz" / f"matches_f{fids[k]:06d}.png"),
                    cv2.resize(canvas, None, fx=0.5, fy=0.5))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
