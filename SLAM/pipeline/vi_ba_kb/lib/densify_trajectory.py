#!/usr/bin/env python3
"""Densify an optimized keyframe trajectory to full frame rate.

Standard pose-graph correction propagation: at each optimized keyframe k,
compute the correction D_k = T_opt_k o inv(T_vio_k). For every full-rate VIO
pose at time t between keyframes k and k+1, apply the interpolated correction
D(t) = interp(D_k, D_{k+1}, u) o T_vio(t), with slerp on rotation and lerp on
translation of the correction, u = (t - t_k) / (t_{k+1} - t_k).

Local VIO accuracy is preserved; global drift is absorbed by the corrections.

Usage: densify_trajectory.py <vio_full.tum> <keyframes_opt.tum> <out.tum>
"""
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


def load_tum(path):
    d = np.loadtxt(path, comments="#")
    return d[np.argsort(d[:, 0])]


def q_mult(a, b):  # xyzw
    x1, y1, z1, w1 = a.T
    x2, y2, z2, w2 = b.T
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=-1)


def q_conj(q):
    out = q.copy()
    out[..., :3] *= -1
    return out


def q_rot(q, v):
    t = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:4] * t + np.cross(q[..., :3], t)


def slerp(q0, q1, u):
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d).clip(-1, 1)
    th = np.arccos(d)
    s = np.sin(th)
    small = s[..., 0] < 1e-6
    w0 = np.where(small[..., None], 1 - u, np.sin((1 - u) * th) / np.where(s == 0, 1, s))
    w1 = np.where(small[..., None], u, np.sin(u * th) / np.where(s == 0, 1, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# A correction may differ from its local neighbourhood by at most this much
# before it is treated as a solver failure rather than drift. 3 deg is well
# above the 0.9 deg median correction seen on healthy episodes and well below
# the 43 deg spikes that ruin the interpolation.
ROT_DEV_DEG = 3.0
POS_DEV_M = 0.25
MED_WINDOW = 9


def _local_median(x, w=MED_WINDOW):
    h = w // 2
    return np.array([np.median(x[max(0, k - h):k + h + 1], axis=0)
                     for k in range(len(x))])


NO_GAUGE_FIX = os.environ.get("MECKA_NO_GAUGE_FIX", "0") == "1"
LINEAR_INTERP = os.environ.get("MECKA_LINEAR_INTERP", "0") == "1"


def _log_map(q):
    """quaternion (xyzw) -> rotation vector. Corrections are small, so this is
    well away from the pi wrap."""
    qn = q / np.linalg.norm(q, axis=-1, keepdims=True)
    qn = np.where(qn[:, 3:4] < 0, -qn, qn)       # same hemisphere
    w = np.clip(qn[:, 3], -1, 1)
    s = np.linalg.norm(qn[:, :3], axis=-1)
    ang = 2 * np.arctan2(s, w)
    k = np.where(s < 1e-12, 0.0, ang / np.where(s < 1e-12, 1, s))
    return qn[:, :3] * k[:, None]


def _exp_map(rv):
    ang = np.linalg.norm(rv, axis=-1)
    half = ang / 2
    s = np.where(ang < 1e-12, 0.5, np.sin(half) / np.where(ang < 1e-12, 1, ang))
    return np.column_stack([rv * s[:, None], np.cos(half)])


def _drop_global_rotation(q_d):
    """Remove the constant rotation common to every correction.

    The refinement BA's global ORIENTATION is not something it can determine.
    Yaw about gravity is unobservable in monocular VI (there is no compass), and
    roll/pitch is only as good as the gravity direction it was handed -- which
    on a noisy front-end is a couple of degrees out. So the solver is free to
    rotate its entire solution, and it does.

    Measured on 6a7d0bc7: the corrections were a 3.10 deg GLOBAL rotation plus
    0.59 deg of genuine per-pose refinement. The global part is the part the BA
    could not have known, and applying it moved attitude_agreement from 0.31%
    to 10.81% over limit -- while the input's own attitude tracked Apple's
    filter to 1.012 deg. The input's gauge is simply better constrained than
    the BA's, so keep it and take only what the BA actually learned: the local
    deviations.

    Implemented as the chordal (projected-arithmetic) mean of the corrections,
    which is the right notion of "average rotation" here because the spread is
    small.
    """
    R = Rotation.from_quat(q_d)
    U, _, Vt = np.linalg.svd(R.as_matrix().mean(axis=0))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R_glob = Rotation.from_matrix(U @ S @ Vt)
    ang = R_glob.magnitude() * 180 / np.pi
    if ang < 1e-3:
        return q_d
    print(f"note: removed a {ang:.2f} deg global rotation from the correction "
          f"field (BA gauge freedom, not a measurement)")
    return (R_glob.inv() * R).as_quat()


def _inlier_corrections(q_d, p_d):
    """Flag keyframe corrections that depart from their local neighbourhood."""
    # rotation as a vector so it can be median-filtered componentwise; the
    # corrections are small, so the small-angle map is faithful here
    qn = q_d / np.linalg.norm(q_d, axis=-1, keepdims=True)
    w = np.clip(qn[:, 3], -1, 1)
    ang = 2 * np.arccos(np.abs(w))
    s = np.linalg.norm(qn[:, :3], axis=-1)
    axis = qn[:, :3] / np.where(s < 1e-12, 1, s)[:, None]
    rv = axis * np.where(s < 1e-12, 0, ang)[:, None] * np.sign(w)[:, None]

    dev_r = np.linalg.norm(rv - _local_median(rv), axis=1) * 180 / np.pi
    dev_p = np.linalg.norm(p_d - _local_median(p_d), axis=1)
    return (dev_r < ROT_DEV_DEG) & (dev_p < POS_DEV_M)


def main():
    vio = load_tum(sys.argv[1])
    opt = load_tum(sys.argv[2])

    # associate keyframes to vio poses by nearest timestamp
    idx = np.searchsorted(vio[:, 0], opt[:, 0])
    idx = np.clip(idx, 0, len(vio) - 1)
    for i, k in enumerate(idx):
        if k > 0 and abs(vio[k - 1, 0] - opt[i, 0]) < abs(vio[k, 0] - opt[i, 0]):
            idx[i] = k - 1
    dt = np.abs(vio[idx, 0] - opt[:, 0])
    # tolerance: 1.5x the VIO frame interval; drop unmatched keyframes rather
    # than fail (the filter occasionally skips frames near keyframes)
    tol = 1.5 * np.median(np.diff(vio[:, 0]))
    good = dt <= tol
    if good.sum() < 0.5 * len(opt):
        raise RuntimeError(
            f"only {good.sum()}/{len(opt)} keyframes associate within "
            f"{tol*1e3:.0f} ms - trajectories do not correspond")
    if (~good).any():
        print(f"note: dropped {(~good).sum()}/{len(opt)} keyframes beyond "
              f"{tol*1e3:.0f} ms association tolerance")
    opt = opt[good]
    idx = idx[good]

    # corrections at keyframes: D = T_opt o inv(T_vio)
    p_v, q_v = vio[idx, 1:4], vio[idx, 4:8]
    p_o, q_o = opt[:, 1:4], opt[:, 4:8]
    q_d = q_mult(q_o, q_conj(q_v))
    p_d = p_o - q_rot(q_d, p_v)

    # Reject outlier corrections. The correction field absorbs DRIFT, so it has
    # to vary slowly along the trajectory; a keyframe whose correction jumps
    # away from its neighbours and back is the bundle adjuster failing on that
    # one pose, not drift. Interpolating through such a spike smears it over the
    # whole segment: on 6a7d0bc7 a single keyframe at t=39.8s moved 43 deg and
    # 978 mm, and dropping just that one took the densified QA from 48 to 62.
    if not NO_GAUGE_FIX:
        q_d = _drop_global_rotation(q_d)

    keep = _inlier_corrections(q_d, p_d)
    if not keep.all():
        print(f"note: dropped {(~keep).sum()}/{len(keep)} outlier corrections "
              f"(>{ROT_DEV_DEG}deg or >{POS_DEV_M*1e3:.0f}mm from local median)")
        if keep.sum() < 2:
            raise RuntimeError("fewer than 2 usable keyframe corrections")
        opt, q_d, p_d = opt[keep], q_d[keep], p_d[keep]

    # Interpolate the correction for every vio pose.
    #
    # NOT linear slerp. Linear interpolation is C0: its derivative is constant
    # inside a segment and JUMPS at every keyframe. Angular velocity is that
    # derivative, so a linearly-interpolated correction injects a step into the
    # rate at each of the ~400 keyframes -- one every 0.267 s -- and
    # gyro_consistency measures exactly that quantity against the real gyro.
    # The correction field is drift: physically smooth, with no reason to be
    # kinked. PCHIP is C1 and shape-preserving, so the rate stays continuous
    # without overshooting near a correction that moves quickly.
    t = vio[:, 0]
    tk = opt[:, 0]
    if len(tk) >= 3 and not LINEAR_INTERP:
        rv = _log_map(q_d)                       # corrections are small
        rv_i = np.stack([PchipInterpolator(tk, rv[:, i])(t) for i in range(3)], 1)
        qc = _exp_map(rv_i)
        pc = np.stack([PchipInterpolator(tk, p_d[:, i])(t) for i in range(3)], 1)
    else:
        seg = np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2)
        u = ((t - tk[seg]) / (tk[seg + 1] - tk[seg])).clip(0, 1)[:, None]
        qc = slerp(q_d[seg], q_d[seg + 1], u)
        pc = (1 - u) * p_d[seg] + u * p_d[seg + 1]

    q_out = q_mult(qc, vio[:, 4:8])
    p_out = q_rot(qc, vio[:, 1:4]) + pc

    with open(sys.argv[3], "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.9f} " +
                    " ".join(f"{v:.9f}" for v in p_out[i]) + " " +
                    " ".join(f"{v:.9f}" for v in q_out[i]) + "\n")
    print(f"wrote {sys.argv[3]}: {len(t)} poses "
          f"(corrected against {len(opt)} keyframes)")


if __name__ == "__main__":
    main()
