#!/usr/bin/env python3
"""Per-keyframe roll/pitch priors from CoreMotion attitude, for the VI-BA.

The solver already implements RollPitchFactor (Basalt's rollPitchError) but we
have never fed it any. Today the gravity direction enters only once, as a single
global rotation applied to the whole model; every per-frame attitude estimate
Apple produces is thrown away. CoreMotion fuses gyro + accel + magnetometer and
keeps tilt bounded, so per-keyframe roll/pitch is genuinely independent
information -- unlike a pose prior taken from the reconstruction we are refining,
which only tells the solver what it already believes.

Residual the solver computes (verified against vi_bundle_adjuster.cc):

    v = (q_w_i_meas * q_IMU_CAM) * (q_cam_from_world * (0,0,-1))
    r = sqrt_info * v.head<2>()

i.e. take the world gravity direction, express it in the body frame using the
ESTIMATED pose, rotate it back to world with the MEASURED attitude, and read off
how far it has tipped off -Z. Yaw-invariant by construction, which is the whole
point -- CoreMotion's heading is not trustworthy but its tilt is.

CoreMotion's reference frame is verified (not assumed) to have gravity along -Z,
matching the gravity-aligned world make_imu_factors builds.

The information matrix is measured, not chosen: we evaluate the residual at the
input model and set sigma from its robust spread (MAD). Picking a number here
would repeat the mistake the IMU preintegration block already makes, where it
claims 0.2 mm position precision against a real ~8 mm disagreement.

Usage: make_rollpitch_factors.py <ep> <model_dir> <out_factors.txt>
                                 [--sigma-scale K] [--report out.json]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

P1 = Path(os.environ.get("MECKA_ROOT", "/pipeline/data"))
# camera -> IMU(body); identical to the solver's T_IMU_CAM and vi_align's R_bc
R_BC = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
T_BC = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])
Q_BC_XYZW = (0.7071068, -0.7071068, 0.0, 0.0)


def read_keyframes(model_dir):
    """[(t_ns, R_cam_from_world)] from a model whose images are named <t_ns>.jpg"""
    # Keep BLANK lines. images.txt is strictly two lines per image, and an
    # image with no observations has an empty second line -- dropping it slides
    # every later pair by one, so POINTS2D rows get parsed as headers and real
    # headers get skipped. That produced 503 factors from a 515-image model,
    # 68 of them with garbage timestamps, and a bogus "keyframes fall outside
    # the attitude stream" warning (the stream covers all of them).
    L = [l for l in open(Path(model_dir) / "images.txt")
         if not l.startswith("#")]
    out = []
    for i in range(0, len(L) - 1, 2):
        h = L[i].split()
        if len(h) < 10:
            continue
        qw, qx, qy, qz = map(float, h[1:5])
        R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        out.append((int(Path(h[9]).stem), R_cw))
    out.sort(key=lambda r: r[0])
    return out


def attitude_from_raw_accel(sensor, t_kf, smooth_s=0.5):
    """Roll/pitch per keyframe from the RAW accelerometer alone.

    Fallback for episodes where Mecka has purged the processed streams --
    episode_meta.json lists processed_orientation.csv and processed_gravity.csv
    under storagePurged, and 18 of the 200 seed episodes have neither. Without
    this the roll-pitch block is empty and VI-BA runs on rel-pose and IMU
    preintegration only, which are both RELATIVE: they tie neighbours together
    and leave the trajectory's absolute attitude free to drift.

    Why not reuse vi_align's gravity estimate instead: it is solved jointly with
    the trajectory, so constraining that trajectory's attitude with it is
    circular -- the factors would be satisfied by construction and anchor
    nothing. A raw accelerometer reading is an INDEPENDENT observation of
    gravity in the body frame, which is exactly what is needed here.

    Sign convention is measured, not assumed. Against episodes that carry both
    streams (d982, d99e, d97d):

        dot(-acc_hat, gravity_hat) = 0.9975 / 0.9996 / 0.9986
        angle between them, median = 4.07 / 1.54 / 3.01 deg

    so raw acc is the negation of CoreMotion's gravity vector, and the residual
    few degrees is linear acceleration, which the smoothing below suppresses.

    Yaw is unrecoverable from gravity and is left arbitrary -- correctly, since
    the solver's residual uses only the x,y components of world gravity and is
    invariant to rotation about z.
    """
    acc = pd.read_csv(sensor / "raw_accelerometer.csv")
    t_a = acc["timestamp"].to_numpy()
    A = acc[["acc_x", "acc_y", "acc_z"]].to_numpy()

    # Boxcar over ~0.5 s. Gravity is quasi-static while hand motion is not, so
    # averaging pulls the estimate towards gravity; the leftover error inflates
    # the MAD-derived sigma below, which is exactly how it should be weighted.
    n = len(t_a)
    if n > 1:
        dt = float(np.median(np.diff(t_a)))
        w = max(1, round(smooth_s / dt) | 1)          # odd window
        pad = w // 2
        ker = np.ones(w) / w
        A = np.stack([np.convolve(np.pad(A[:, i], pad, mode="edge"), ker,
                                  mode="valid") for i in range(3)], 1)

    g_body = -A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    g_kf = np.stack([np.interp(np.clip(t_kf, t_a[0], t_a[-1]), t_a, g_body[:, i])
                     for i in range(3)], 1)
    g_kf /= np.maximum(np.linalg.norm(g_kf, axis=1, keepdims=True), 1e-9)

    # minimal rotation carrying measured body gravity onto world -Z
    down = np.array([0.0, 0.0, -1.0])
    R_list = []
    for u in g_kf:
        v = np.cross(u, down)
        c = float(np.dot(u, down))
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            R_list.append(np.eye(3) if c > 0 else
                          np.diag([1.0, -1.0, -1.0]))
            continue
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_list.append(np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2))
    return Rotation.from_matrix(np.array(R_list)), t_a


def main():
    ep, model_dir, out_path = sys.argv[1:4]
    k = float(sys.argv[sys.argv.index("--sigma-scale") + 1]) \
        if "--sigma-scale" in sys.argv else 1.0

    kf = read_keyframes(model_dir)
    t_kf = np.array([r[0] for r in kf]) / 1e9

    sensor = P1 / "data/episodes" / ep / "sensor"
    orient_f = sensor / "processed_orientation.csv"
    grav_f = sensor / "processed_gravity.csv"
    raw_f = sensor / "raw_accelerometer.csv"

    if orient_f.exists() and grav_f.exists():
        source = "CoreMotion"
        att = pd.read_csv(orient_f)
        t_a = att["timestamp"].to_numpy()
        R_a = Rotation.from_quat(
            att[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy())

        # verify (do not assume) that CoreMotion's reference frame is gravity = -Z
        grav = pd.read_csv(grav_f)
        G = np.stack([np.interp(t_a, grav["timestamp"].to_numpy(),
                                grav[f"gravity_{c}"].to_numpy())
                      for c in "xyz"], 1)
        s = np.arange(0, len(t_a), max(1, len(t_a) // 500))
        v = R_a[s].apply(G[s])
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        up = v.mean(0)
        if up[2] > -0.9:
            print(f"ABORT: CoreMotion reference gravity is {up.round(3)}, "
                  f"not -Z; the factor convention would be wrong")
            return
        slerp = Slerp(t_a, R_a)
        R_wi = slerp(np.clip(t_kf, t_a[0], t_a[-1]))
        print(f"  CoreMotion ref gravity {up.round(4)}  (verified -Z)")
    elif raw_f.exists():
        # Degrade, do not die. This used to be an unguarded read_csv, so a
        # purged stream raised FileNotFoundError, the pipeline logged it as a
        # failed stage, and the episode was refined with NO absolute attitude
        # constraint at all -- reported only as "0 roll-pitch factors".
        source = "raw accelerometer"
        R_wi, t_a = attitude_from_raw_accel(sensor, t_kf)
        print("  processed_orientation/gravity absent (purged upstream); "
              "roll-pitch derived from raw_accelerometer")
    else:
        print("SKIP: no attitude source (processed_orientation.csv, "
              "processed_gravity.csv and raw_accelerometer.csv all absent) -- "
              "writing no roll-pitch factors")
        return

    inside = (t_kf >= t_a[0]) & (t_kf <= t_a[-1])

    # residual at the input model -> this is what sets sigma
    res = []
    for i, (_, R_cw) in enumerate(kf):
        g_body = R_BC @ (R_cw @ np.array([0.0, 0.0, -1.0]))
        res.append((R_wi[i].as_matrix() @ g_body)[:2])
    res = np.array(res)

    # robust spread: MAD -> sigma, per axis, so a single bad stretch cannot
    # inflate the weight the whole episode is given
    med = np.median(res, axis=0)
    mad = np.median(np.abs(res - med), axis=0)
    sigma = np.maximum(1.4826 * mad, 1e-6) * k
    info = np.diag(1.0 / sigma ** 2)

    n_out = int((~inside).sum())
    with open(out_path, "w") as f:
        f.write("T_IMU_CAM " + " ".join(f"{x:.17g}" for x in T_BC) + " " +
                " ".join(f"{x:.7g}" for x in Q_BC_XYZW) + "\n")
        f.write("RELPOSE 0\n")
        f.write(f"ROLLPITCH {len(kf)}\n")
        for i, (t_ns, _) in enumerate(kf):
            q = R_wi[i].as_quat()          # xyzw, which is what the solver reads
            f.write(f"{t_ns} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                    + " ".join(f"{x:.9e}" for x in info.reshape(-1)) + "\n")

    tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(res, axis=1), -1, 1)))
    print(f"{len(kf)} roll-pitch factors -> {out_path}  [source: {source}]")
    print(f"  tilt disagreement vs input model: median {np.median(tilt):.3f} deg,"
          f" p95 {np.percentile(tilt, 95):.3f} deg, max {tilt.max():.3f} deg")
    print(f"  measured sigma (MAD) = {sigma.round(6)}  -> info diag "
          f"{np.diag(info).round(1)}   [sigma-scale {k:g}]")
    if n_out:
        print(f"  NOTE: {n_out} keyframes fall outside the attitude stream and "
              f"were clamped to its endpoints")
    if "--report" in sys.argv:
        json.dump({"episode": ep, "n_factors": len(kf),
                   "tilt_median_deg": float(np.median(tilt)),
                   "tilt_p95_deg": float(np.percentile(tilt, 95)),
                   "sigma": sigma.tolist(), "sigma_scale": k},
                  open(sys.argv[sys.argv.index("--report") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
