#!/usr/bin/env python3
"""Relative-pose factors for colmap_vi_ba, from the front-end's own trajectory.

This is the constraint that stops the refinement BA from wandering, and it is
the one thing Project1's delivered chain had that ours did not. Its
synth_factors_from_vio.py put it plainly: "the factors keep the BA from
collapsing (the Project-2 lesson)."

Why it matters here: for OpenVINS and DROID the structure is triangulated FROM
the poses being refined, so the visual residual alone cannot tell the solver
where the cameras really are -- it will happily move them to fit noisy points.
IMU preintegration constrains inertial consistency but not "stay near what the
front-end measured". Without a rel-pose tie the BA improved reprojection while
adding rotation noise, and the refined result scored BELOW the raw one
(DROID on 6a7d0bc7: raw 94.0, refined 76.8).

VIO/DROID odometry is locally accurate -- that is precisely the thing it is
good at -- so a relative-pose prior between consecutive keyframes is cheap,
independent of the structure, and exactly the right shape of constraint. Loop
closures still come from COLMAP's matches; this only pins the local motion.

Keys factors by the model's image names, which by this point in the pipeline
are nanosecond timestamps (make_imu_factors renames them), so the solver's
stem-to-int mapping lines up with the IMUPREINT block.

Usage: make_relpose_factors.py <traj.tum> <model_dir> <out_factors.txt>
                               [--sigma-t M] [--sigma-r RAD] [--max-dt S]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

R_bc = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
t_bc = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])

# calibrated by Project1 as the p90 of factor residuals at the BA optimum
SIGMA_T = 0.010     # m,   over a consecutive-keyframe interval
SIGMA_R = 0.005     # rad, over the same


def R_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = np.zeros(4)
        q[1 + i] = s / 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def quat_xyzw_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--sigma-t", type=float, default=SIGMA_T)
    ap.add_argument("--sigma-r", type=float, default=SIGMA_R)
    ap.add_argument("--max-dt", type=float, default=0.05)
    ap.add_argument("--rollpitch", default=None,
                    help="a ROLLPITCH factor file to merge in. The solver reads "
                         "T_IMU_CAM, RELPOSE and ROLLPITCH from ONE file, and "
                         "rel-pose alone cannot stop slow drift -- it only ties "
                         "neighbours, so error accumulating over hundreds of "
                         "frames costs almost nothing per pair. Roll-pitch is "
                         "the absolute constraint (84 -> 91 on 6a7d0bc7).")
    a = ap.parse_args()

    d = np.loadtxt(a.traj, comments="#")
    t_v, p_v, q_v = d[:, 0], d[:, 1:4], d[:, 4:8]

    # registered images, keyed by whatever integer their name is
    keys = []
    for line in open(Path(a.model) / "images.txt"):
        f = line.strip().split()
        if len(f) == 10 and f[-1].endswith(".jpg"):
            try:
                keys.append(int(Path(f[-1]).stem))
            except ValueError:
                continue
    keys.sort()
    if len(keys) < 2:
        raise SystemExit(f"{a.model}: fewer than 2 registered images")

    # names are nanosecond timestamps by this stage of the pipeline
    scale = 1e-9 if keys[-1] > 10**12 else 1.0
    if scale == 1.0:
        raise SystemExit(
            "model images are not named by nanosecond timestamp -- run this "
            "on the model make_imu_factors prepared, not the raw one")

    poses, dropped = {}, 0
    for k in keys:
        t = k * scale
        j = int(np.argmin(np.abs(t_v - t)))
        if abs(t_v[j] - t) > a.max_dt:
            dropped += 1
            continue
        T = np.eye(4)
        T[:3, :3] = quat_xyzw_to_R(*q_v[j])
        T[:3, 3] = p_v[j]
        poses[k] = T                      # world_from_body

    reg = sorted(poses)
    if len(reg) < 2:
        raise SystemExit("no consecutive registered pairs to constrain")

    info = np.diag([1 / a.sigma_t**2] * 3 + [1 / a.sigma_r**2] * 3)
    info_flat = " ".join(f"{v:.9e}" for v in info.reshape(-1))

    rows = []
    for i, j in zip(reg[:-1], reg[1:]):
        # T_i_j maps body_j into body_i, which is what the solver expects
        T_ij = np.linalg.inv(poses[i]) @ poses[j]
        q = R_to_quat_wxyz(T_ij[:3, :3])          # wxyz
        t = T_ij[:3, 3]
        rows.append(f"{i} {j} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {q[0]:.12f} "
                    f"{info_flat}")

    rp_rows = []
    if a.rollpitch and Path(a.rollpitch).exists():
        txt = open(a.rollpitch).read()
        if "ROLLPITCH" in txt:
            body = txt.split("ROLLPITCH", 1)[1].strip().splitlines()
            rp_rows = [ln for ln in body[1:] if ln.strip()]

    q_ic = R_to_quat_wxyz(R_bc)
    with open(a.out, "w") as f:
        f.write(f"T_IMU_CAM {t_bc[0]} {t_bc[1]} {t_bc[2]} "
                f"{q_ic[1]} {q_ic[2]} {q_ic[3]} {q_ic[0]}\n")
        f.write(f"RELPOSE {len(rows)}\n")
        for r in rows:
            f.write(r + "\n")
        f.write(f"ROLLPITCH {len(rp_rows)}\n")
        for r in rp_rows:
            f.write(r + "\n")

    print(f"wrote {len(rows)} rel-pose + {len(rp_rows)} roll-pitch factors "
          f"from {len(reg)}/{len(keys)} registered images -> {a.out}"
          + (f"  ({dropped} had no pose within {a.max_dt*1e3:.0f} ms)"
             if dropped else ""))


if __name__ == "__main__":
    sys.exit(main())
