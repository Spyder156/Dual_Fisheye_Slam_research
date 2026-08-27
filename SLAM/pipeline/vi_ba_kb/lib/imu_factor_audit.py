#!/usr/bin/env python3
"""Per-block audit of the IMU preintegration factors against a COLMAP model.

Answers "which of the nine residual dimensions is blowing up the BA", which
the position-only check cannot. Evaluates the true Forster residual at the
INPUT geometry, so anything large here is a factor/model disagreement that
exists before the solver takes a single step.

    r_rot = Log( dR_meas^T  R_i^T R_j )
    r_vel = R_i^T ( v_j - v_i - g dt )        - dv_meas
    r_pos = R_i^T ( p_j - p_i - v_i dt - .5 g dt^2 ) - dp_meas

    cost  = 0.5 * r^T Lambda r          (Lambda = the stored information matrix)

Reports the cost split by block using the full quadratic form, not just the
diagonal -- an ill-conditioned Lambda can make the true cost far larger than a
diagonal reading suggests.

Poses come from the model (camera, world->cam) and are converted to body with
the same R_bc/t_bc the aligner and the C++ solver use. Velocities come from the
IMUINIT block of the factor file.

Usage: imu_factor_audit.py <model_dir> <factors.txt> [--json out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

R_BC = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
T_BC = np.array([0.033366085092802436, 0.009419070514053628,
                 -0.006188374507046947])
G_W = np.array([0.0, 0.0, -9.805])


def q2R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def logSO3(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    return th / (2 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def read_body_poses(model_dir):
    """image-stem -> (p_wb, R_wb), using the shared camera->body extrinsic."""
    L = [l for l in open(Path(model_dir) / "images.txt")
         if not l.startswith("#") and l.strip()]
    out = {}
    for i in range(0, len(L) - 1, 2):
        h = L[i].split()
        if len(h) < 10:
            continue
        R_cw = q2R(*map(float, h[1:5]))
        t_cw = np.array(list(map(float, h[5:8])))
        R_wc = R_cw.T
        p_wc = -R_cw.T @ t_cw
        R_wb = R_wc @ R_BC.T
        out[int(Path(h[9]).stem)] = (p_wc - R_wb @ T_BC, R_wb)
    return out


def main():
    model_dir, factors = sys.argv[1], sys.argv[2]
    pose = read_body_poses(model_dir)

    txt = open(factors).read().split("\n")
    n = int(txt[0].split()[1])
    vel = {}
    for i, l in enumerate(txt):
        if l.startswith("IMUINIT"):
            for row in txt[i + 1:i + 1 + int(l.split()[1])]:
                f = row.split()
                if len(f) == 4:
                    vel[int(f[0])] = np.array(f[1:4], float)
            break

    rows = []
    for l in txt[1:1 + n]:
        f = l.split()
        if len(f) < 90:
            continue
        ta, tb, dt = int(f[0]), int(f[1]), float(f[2])
        # make_imu_factors' q_from_R emits (x, y, z, w); q2R takes (w, x, y, z)
        qx, qy, qz, qw = map(float, f[3:7])
        dR = q2R(qw, qx, qy, qz)
        dv = np.array(f[7:10], float)
        dp = np.array(f[10:13], float)
        Lam = np.array(f[-81:], float).reshape(9, 9)
        if ta not in pose or tb not in pose or ta not in vel or tb not in vel:
            continue
        pi, Ri = pose[ta]
        pj, Rj = pose[tb]
        vi, vj = vel[ta], vel[tb]
        r = np.concatenate([
            logSO3(dR.T @ (Ri.T @ Rj)),
            Ri.T @ (vj - vi - G_W * dt) - dv,
            Ri.T @ (pj - pi - vi * dt - 0.5 * G_W * dt * dt) - dp,
        ])
        Lam = 0.5 * (Lam + Lam.T)
        # cost contribution of each block including its cross terms, so the
        # three numbers sum to the total quadratic form
        cost = 0.5 * float(r @ Lam @ r)
        blocks = [0.5 * float(r[s] @ Lam[s][:, s] @ r[s])
                  for s in (slice(0, 3), slice(3, 6), slice(6, 9))]
        ev = np.linalg.eigvalsh(Lam)
        rows.append({"ta": ta, "dt": dt, "cost": cost, "rot": blocks[0], "vel": blocks[1],
                         "pos": blocks[2],
                         "r_rot": float(np.linalg.norm(r[0:3])),
                         "r_vel": float(np.linalg.norm(r[3:6])),
                         "r_pos": float(np.linalg.norm(r[6:9])),
                         "emin": float(ev[0]), "emax": float(ev[-1])})

    if not rows:
        print("no factors matched the model -- check the name/timestamp keys")
        return
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    tot = A["cost"].sum()
    print(f"{len(rows)} factors matched, dt median {np.median(A['dt']):.3f} s")
    print(f"total IMU cost at the input geometry : {tot:.4e}")
    print(f"  rot block  {A['rot'].sum():.4e}  ({100*A['rot'].sum()/tot:5.1f}%)"
          f"   |r| median {np.degrees(np.median(A['r_rot'])):.3f} deg")
    print(f"  vel block  {A['vel'].sum():.4e}  ({100*A['vel'].sum()/tot:5.1f}%)"
          f"   |r| median {np.median(A['r_vel'])*1000:.2f} mm/s")
    print(f"  pos block  {A['pos'].sum():.4e}  ({100*A['pos'].sum()/tot:5.1f}%)"
          f"   |r| median {np.median(A['r_pos'])*1000:.3f} mm")
    print(f"worst factor cost {A['cost'].max():.3e} at t={int(A['ta'][A['cost'].argmax()])}")
    print(f"Lambda eigenvalues: min median {np.median(A['emin']):.3e}, "
          f"max median {np.median(A['emax']):.3e}, "
          f"condition median {np.median(A['emax']/np.maximum(A['emin'],1e-30)):.3e}")
    if "--json" in sys.argv:
        json.dump({k: A[k].tolist() for k in A},
                  open(sys.argv[sys.argv.index("--json") + 1], "w"))


if __name__ == "__main__":
    main()
