#!/usr/bin/env python3
"""Write IMU preintegration factors for colmap_vi_ba.

For each consecutive keyframe pair: preintegrate gyro+accel (midpoint), with
bias Jacobians and a propagated covariance, and emit the IMUPREINT block the
C++ solver reads.

Also writes the gravity-aligned, metric-scaled COLMAP model the solver should
start from (world rotated so gravity is -Z, geometry scaled by the linear
alignment's s), because the C++ factors assume g = (0,0,-|g|).

Usage: make_imu_factors.py <ep> <model_in> <align_json> <out_model> <out_factors>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P1 = Path(os.environ.get("MECKA_ROOT", "/pipeline/data"))
SIG_A = 2.4e-2      # accel noise density  [m/s^2/sqrt(Hz)]  (measured)
SIG_G = 5.0e-4      # gyro noise density   [rad/s/sqrt(Hz)]  (measured)
MAX_DT = 0.40       # skip keyframe gaps longer than this (gravity integrates
                    # too far; these are dropped-frame holes, not real motion)


# KB variant: our rig's imu_from_cam, from the kalibr chain (env-overridable).
def _load_R_BC():
    import re as _re, os as _os
    p = _os.environ.get("KB_IMUCAM", "")
    if not p or not _os.path.exists(p):
        raise SystemExit("set KB_IMUCAM=<kalibr_imucam_chain.yaml>")
    blk = _re.split(r"\ncam\d+:", open(p).read())[1]
    n = [float(x) for x in _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                       blk.split("cam_overlaps")[0])]
    return np.array(n).reshape(4, 4)[:3, :3]
R_BC = _load_R_BC()   # imu_from_cam


def so3_log(R):
    """rotation matrix -> rotation vector (small angles here)"""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2 * np.sin(th)))


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def q_from_R(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        q = [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], 1 + tr]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        q = [1 + R[0, 0] - R[1, 1] - R[2, 2], R[0, 1] + R[1, 0],
             R[0, 2] + R[2, 0], R[2, 1] - R[1, 2]]
    elif R[1, 1] >= R[2, 2]:
        q = [R[0, 1] + R[1, 0], 1 + R[1, 1] - R[0, 0] - R[2, 2],
             R[1, 2] + R[2, 1], R[0, 2] - R[2, 0]]
    else:
        q = [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1],
             1 + R[2, 2] - R[0, 0] - R[1, 1], R[1, 0] - R[0, 1]]
    q = np.array(q, float)
    return q / np.linalg.norm(q)           # [x, y, z, w]


def main():
    ep, model_in, align_json, out_model, out_factors = sys.argv[1:6]
    al = json.load(open(align_json))
    s = al["scale"]
    if os.environ.get("MECKA_MODEL_PREALIGNED", "0") == "1":
        s = 1.0                       # already metric -- see the note below
    # Is the incoming model ALREADY metric and gravity-aligned?
    #
    # It is, whenever the trajectory it was triangulated from went through
    # apply_scale first -- which is every branch now. Applying the scale and
    # the gravity rotation again here transforms the model a second time: the
    # model handed to the solver came out at 9.95 m against 30.57 m of IMU
    # (3.07x), and the BA "resolved" that by wrecking the poses. Project1 never
    # hit this because its factor writer (synth_factors_from_vio.py) has no
    # scale or gravity logic at all -- the trajectory reaches it already metric
    # and the transform happens exactly once, upstream.
    already = os.environ.get("MECKA_MODEL_PREALIGNED", "0") == "1"  # noqa: F841

    # NOTE what `already` does and does not cover. It means the model is
    # already METRIC -- the scale was applied upstream and must not be applied
    # twice. It does NOT mean gravity is exactly -Z. "Gravity-aligned" upstream
    # is only as good as the estimate that aligned it: on 6a7d0bc7 the model
    # still carried a 3.15 deg residual tilt. Forcing gd to -Z here told the
    # solver the model was already upright, so the solver discovered the tilt
    # itself and rotated the ENTIRE trajectory by 3.10 deg to fix it -- which
    # is precisely the global rotation that wrecked attitude_agreement
    # (0.31% -> 10.81% over limit) while the real per-pose refinement was only
    # 0.68 deg. Use the MEASURED direction so the model is genuinely upright
    # before the solver sees it, and the solver has nothing left to rotate.
    gd = np.array(al["gravity_dir"], float)
    gd /= np.linalg.norm(gd)

    # world rotation putting gravity along -Z
    zt = np.array([0, 0, -1.0])
    v = np.cross(gd, zt)
    c = float(np.dot(gd, zt))
    if np.linalg.norm(v) < 1e-8:
        R_up = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = skew(v)
        R_up = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

    # ---- rewrite the model: gravity-aligned + metric ------------------
    model_in, out_model = Path(model_in), Path(out_model)
    out_model.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_in / "cameras.txt", out_model / "cameras.txt")
    # rigs.txt is only a sensor-layout declaration -- no world pose, safe to
    # copy. frames.txt is NOT: from COLMAP 3.12 on it carries RIG_FROM_WORLD
    # for every frame, and the reader takes the pose from there in preference
    # to images.txt. Copying it verbatim silently reverted every pose to the
    # untransformed model while the points stayed gravity-rotated and
    # metric-scaled, so the BA started from a reconstruction whose cameras and
    # points were in different frames. It must get the same transform.
    if (model_in / "rigs.txt").exists():
        shutil.copy(model_in / "rigs.txt", out_model / "rigs.txt")

    def q2R(x, y, z, w):
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    lines = [l for l in open(model_in / "images.txt") if not l.startswith("#")]
    out_img = []
    # KB: our frames.csv is (frame,t) with t already on the IMU clock -- there
    # is no separate offset file, the extractor put both on one clock.
    _fc = pd.read_csv(os.environ["KB_FRAMES_CSV"])
    _idx = _fc["frame"].to_numpy().astype(int)
    _t = _fc["t"].to_numpy()
    unix = np.zeros(int(_idx.max()) + 1)
    unix[_idx] = _t
    kf = []                                   # (t_ns, name)
    for i in range(0, len(lines) - 1, 2):
        h = lines[i].split()
        if len(h) != 10:
            continue
        qw, qx, qy, qz = map(float, h[1:5])
        n = np.linalg.norm([qw, qx, qy, qz])
        R_cw = q2R(qx/n, qy/n, qz/n, qw/n)
        t_cw = np.array(list(map(float, h[5:8])))
        # world transform: X' = R_up (s X)  =>  cam_from_world' :
        R_cw2 = R_cw @ R_up.T
        t_cw2 = s * t_cw
        q2 = q_from_R(R_cw2)
        idx = int(Path(h[9]).stem)
        if idx >= len(unix):
            continue
        t_ns = round(unix[idx] * 1e9)
        # rename the image to its timestamp-ns so the solver can key on it
        out_img.append(f"{h[0]} {q2[3]} {q2[0]} {q2[1]} {q2[2]} "
                       f"{t_cw2[0]} {t_cw2[1]} {t_cw2[2]} {h[8]} {t_ns}.jpg\n")
        out_img.append(lines[i + 1])
        # keep the body rotation too: the gyro-bias solve below needs the
        # rotation the model actually claims between consecutive keyframes
        kf.append((t_ns, unix[idx], R_cw2.T @ R_BC.T))
    (out_model / "images.txt").write_text("".join(out_img))
    # R_up is a convenience for the SOLVER, which hard-assumes g = (0,0,-|g|).
    # It is not a correction we want to ship: the refined model comes back in
    # this rotated frame while the trajectory it will be densified against is
    # still in the original one, so the rotation leaks into the output and
    # shows up as a global attitude shift. Record it so the export can undo it.
    json.dump({"R_up": R_up.tolist(), "scale": s},
              open(out_model / "world_transform.json", "w"), indent=1)

    # frames.txt: FRAME_ID RIG_ID QW QX QY QZ TX TY TZ NUM_DATA_IDS DATA_IDS...
    # RIG_FROM_WORLD uses the same world->sensor convention as images.txt, so
    # it takes the identical transform.
    if (model_in / "frames.txt").exists():
        fr_out = []
        for l in open(model_in / "frames.txt"):
            if l.startswith("#") or not l.strip():
                fr_out.append(l)
                continue
            f = l.split()
            qw, qx, qy, qz = map(float, f[2:6])
            n = np.linalg.norm([qw, qx, qy, qz])
            R2 = q2R(qx / n, qy / n, qz / n, qw / n) @ R_up.T
            t2 = s * np.array(list(map(float, f[6:9])))
            qq = q_from_R(R2)
            fr_out.append(" ".join([
                f[0], f[1],
                f"{qq[3]:.12f}", f"{qq[0]:.12f}", f"{qq[1]:.12f}", f"{qq[2]:.12f}",
                f"{t2[0]:.9f}", f"{t2[1]:.9f}", f"{t2[2]:.9f}",
                *f[9:],
            ]) + "\n")
        (out_model / "frames.txt").write_text("".join(fr_out))

    pts_out = []
    for l in open(model_in / "points3D.txt"):
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        X = R_up @ (s * np.array([float(f[1]), float(f[2]), float(f[3])]))
        pts_out.append(" ".join([f[0], f"{X[0]:.9f}", f"{X[1]:.9f}", f"{X[2]:.9f}", *f[4:]]) + "\n")
    (out_model / "points3D.txt").write_text("".join(pts_out))
    print(f"model: {len(kf)} images, {len(pts_out)} points -> {out_model}")

    # ---- preintegrate between consecutive keyframes -------------------
    # KB: one imu.csv, gyro and accel already on the same clock and rate.
    _imu = pd.read_csv(os.environ["KB_IMU_CSV"])
    tg = _imu["t"].to_numpy()
    wg = _imu[["gx", "gy", "gz"]].to_numpy()
    ta = tg
    aa = _imu[["ax", "ay", "az"]].to_numpy()
    lo, hi = max(tg[0], ta[0]), min(tg[-1], ta[-1])
    tt = np.arange(lo, hi, 0.005)
    W = np.stack([np.interp(tt, tg, wg[:, i]) for i in range(3)], 1)
    A = np.stack([np.interp(tt, ta, aa[:, i]) for i in range(3)], 1)

    kf.sort()
    out = []
    bias_rows = []      # (J_q_bg, r0) per factor, for the gyro-bias solve
    for i in range(len(kf) - 1):
        t_i_ns, t0, R_wi_i = kf[i]
        t_j_ns, t1, R_wi_j = kf[i + 1]
        dt_total = t1 - t0
        if not (0 < dt_total < MAX_DT):
            continue
        m = (tt >= t0) & (tt < t1)
        if m.sum() < 3:
            continue
        ts, Ws, As = tt[m], W[m], A[m]
        R = np.eye(3)
        dv = np.zeros(3)
        dp = np.zeros(3)
        J_q_bg = np.zeros((3, 3))
        J_v_bg = np.zeros((3, 3))
        J_v_ba = np.zeros((3, 3))
        J_p_bg = np.zeros((3, 3))
        J_p_ba = np.zeros((3, 3))
        P = np.zeros((9, 9))                   # [dp, dv, dq] covariance
        for k in range(len(ts) - 1):
            dtk = ts[k + 1] - ts[k]
            a_k = 0.5 * (As[k] + As[k + 1])
            w_k = 0.5 * (Ws[k] + Ws[k + 1])
            Ra = R @ a_k
            # state propagation
            dp = dp + dv * dtk + 0.5 * Ra * dtk * dtk
            dv = dv + Ra * dtk
            dR_k = so3_exp(w_k * dtk)
            # bias Jacobians
            J_p_ba = J_p_ba + J_v_ba * dtk - 0.5 * R * dtk * dtk
            J_p_bg = J_p_bg + J_v_bg * dtk + 0.5 * R @ skew(a_k) @ J_q_bg * dtk * dtk
            J_v_ba = J_v_ba - R * dtk
            J_v_bg = J_v_bg + R @ skew(a_k) @ J_q_bg * dtk
            J_q_bg = dR_k.T @ J_q_bg - np.eye(3) * dtk
            # covariance propagation (first order)
            F = np.eye(9)
            F[0:3, 3:6] = np.eye(3) * dtk
            F[0:3, 6:9] = -0.5 * R @ skew(a_k) * dtk * dtk
            F[3:6, 6:9] = -R @ skew(a_k) * dtk
            F[6:9, 6:9] = dR_k.T
            G = np.zeros((9, 6))
            G[0:3, 0:3] = 0.5 * R * dtk * dtk
            G[3:6, 0:3] = R * dtk
            G[6:9, 3:6] = np.eye(3) * dtk
            Qd = np.diag([SIG_A**2 / dtk] * 3 + [SIG_G**2 / dtk] * 3)
            P = F @ P @ F.T + G @ Qd @ G.T
            R = R @ dR_k
        info = np.linalg.inv(P + np.eye(9) * 1e-12)
        q = q_from_R(R)
        row = [str(t_i_ns), str(t_j_ns), f"{dt_total:.9f}",
               *[f"{x:.9f}" for x in q],
               *[f"{x:.9f}" for x in dv], *[f"{x:.9f}" for x in dp]]
        # residual of THIS factor at the current (zero) gyro bias: the
        # rotation the model claims minus the one the gyro preintegrated
        bias_rows.append((J_q_bg.copy(),
                          so3_log(R.T @ (R_wi_i.T @ R_wi_j))))
        for M in (J_q_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba):
            row += [f"{x:.9f}" for x in M.reshape(-1)]
        row += [f"{x:.9e}" for x in info.reshape(-1)]
        out.append(" ".join(row))
    # initial velocities (rotated into the gravity-aligned world, scaled) and
    # accel bias, both from the linear alignment -> solver init
    vel_csv = Path(align_json).with_name(Path(align_json).stem + "_velocities.csv")
    init_rows = []
    if vel_csv.exists():
        V = np.loadtxt(vel_csv, delimiter=",", skiprows=1)
        # interpolate the linear solution's velocities onto THIS model's
        # keyframe times (the two runs need not share timestamps)
        for t_ns, t_unix, _ in kf:
            vv = np.array([np.interp(t_unix, V[:, 0], V[:, c]) for c in (1, 2, 3)])
            # v comes out of the linear alignment already in m/s -- its row
            # block reads Rt(s*dp_world - v*dt - .5 g dt^2) = dp, where dp and
            # g are metric, so v must be metric too. Scaling by s again here
            # was shrinking the solver's velocity init by 1/s.
            vw = R_up @ vv
            init_rows.append(f"{t_ns} {vw[0]:.9f} {vw[1]:.9f} {vw[2]:.9f}")
    # ---- gyro bias --------------------------------------------------------
    # We preintegrate at zero gyro bias, which is the convention the solver
    # expects (it corrects with J_q_bg * bg, using the ABSOLUTE bias state, not
    # a delta from a linearisation point). But we were then handing it bg = 0
    # as the starting estimate while the device's real bias is ~0.8 deg/s.
    #
    # Over a 0.2 s factor that is a 0.15 deg rotation error, present in EVERY
    # factor with a consistent sign -- so the solver saw a coherent rotation
    # drift across all 514 factors and dutifully rotated the trajectory to
    # chase it. That is why more iterations made the result worse, why more
    # keyframes made it worse, and why raising the inertial weight moved the
    # poses further: every one of those trusts the bad factors more.
    #
    # Solve it here instead: the residual at the starting point is
    # r = Log(dq^-1 R_i^T R_j), and the solver models it as J_q_bg * bg, so a
    # least-squares fit over all factors recovers the bias directly.
    bg0 = np.zeros(3)
    if bias_rows:
        A_ = np.vstack([J for J, _ in bias_rows])
        b_ = np.concatenate([r for _, r in bias_rows])
        bg0 = np.linalg.lstsq(A_, b_, rcond=None)[0]
        pred = A_ @ bg0
        before = np.linalg.norm(b_) / np.sqrt(len(b_) / 3)
        after = np.linalg.norm(b_ - pred) / np.sqrt(len(b_) / 3)
        print(f"  gyro bias solved from {len(bias_rows)} factors: "
              f"{np.round(bg0 * 180 / np.pi, 4)} deg/s "
              f"(|b| = {np.linalg.norm(bg0) * 180 / np.pi:.4f})")
        print(f"  rotation residual: {before * 180 / np.pi:.4f} -> "
              f"{after * 180 / np.pi:.4f} deg with it applied")

    ba0 = np.array(al.get("accel_bias", [0, 0, 0]), float)
    with open(out_factors, "w") as f:
        f.write(f"IMUPREINT {len(out)}\n")
        f.write("\n".join(out) + "\n")
        f.write(f"IMUINIT {len(init_rows)} {ba0[0]:.9f} {ba0[1]:.9f} "
                f"{ba0[2]:.9f} {bg0[0]:.9f} {bg0[1]:.9f} {bg0[2]:.9f}\n")
        if init_rows:
            f.write("\n".join(init_rows) + "\n")
    print(f"wrote {len(out)} IMU factors (dt<={MAX_DT}s) + "
          f"{len(init_rows)} velocity inits -> {out_factors}")


if __name__ == "__main__":
    main()
