#!/usr/bin/env python
"""Fit Mei(UCM+radtan) / EUCM / Double-Sphere to a reference KB4 (equidistant) model.

Samples bearings across the FOV, projects them through the reference model, and
least-squares fits each candidate to reproduce those pixels. Initialisation is
derived analytically from the KB4 focal so the solver starts close.

The Hilti calibration ships official EUCM values, so fitting EUCM from KB4 and
comparing against them validates this script end to end.

Usage: fit_camera_models.py [--fov-deg 90] [--json out.json]
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares

# Official Hilti-Trimble calibration
KB4 = {
    "cam0": [465.3015482593691, 465.32303798346413, 730.0455886686005, 720.1427007671206,
             0.025800718903376804, -0.010909240777406872, -0.0016899537986031076, 0.00014766801645260894],
    "cam1": [465.43630225493564, 465.4719940813323, 733.436396975063, 718.7408714249976,
             0.024723386256038673, -0.010754674082879279, -0.0016168993760219885, 0.00013675348254771203],
}
EUCM_OFFICIAL = {
    "cam0": [0.6899954350657926, 0.8911981210457725, 465.2979536302252, 465.3194431883040,
             730.0455886686005, 720.14270076712060],
    "cam1": [0.689023978287534, 0.8955908238645756, 465.43282715729856, 465.46851871720855,
             733.4363969750630, 718.74087142499763],
}


def kb4_project(b, p):
    fx, fy, cx, cy, k1, k2, k3, k4 = p
    x, y, z = b[:, 0], b[:, 1], b[:, 2]
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(r, z)
    d = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    s = np.where(r > 1e-12, d / np.maximum(r, 1e-12), 1.0)
    return np.stack([fx * x * s + cx, fy * y * s + cy], 1)


def mei_project(b, p):
    """UCM + radial-tangential. p = [xi, fx, fy, cx, cy, k1, k2, p1, p2]"""
    xi, fx, fy, cx, cy, k1, k2, p1, p2 = p
    n = np.linalg.norm(b, axis=1, keepdims=True)
    Xs = b / n
    den = Xs[:, 2] + xi
    x, y = Xs[:, 0] / den, Xs[:, 1] / den
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def eucm_project(b, p):
    """p = [alpha, beta, fx, fy, cx, cy]"""
    a, be, fx, fy, cx, cy = p
    x, y, z = b[:, 0], b[:, 1], b[:, 2]
    d = np.sqrt(be * (x * x + y * y) + z * z)
    den = a * d + (1 - a) * z
    return np.stack([fx * x / den + cx, fy * y / den + cy], 1)


def ds_project(b, p):
    """Double Sphere. p = [xi, alpha, fx, fy, cx, cy]"""
    xi, a, fx, fy, cx, cy = p
    x, y, z = b[:, 0], b[:, 1], b[:, 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    zs = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zs * zs)
    den = a * d2 + (1 - a) * zs
    return np.stack([fx * x / den + cx, fy * y / den + cy], 1)


def sample_bearings(fov_deg, n_rad=90, n_az=72):
    """Bearings on a polar grid out to fov_deg, area-weighted so the periphery
    is not under-represented (equal-area in cos(theta))."""
    th_max = np.radians(fov_deg)
    cos_grid = np.linspace(1.0, np.cos(th_max), n_rad)
    th = np.arccos(cos_grid)
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    T, A = np.meshgrid(th, az, indexing="ij")
    T, A = T.ravel(), A.ravel()
    return np.stack([np.sin(T) * np.cos(A), np.sin(T) * np.sin(A), np.cos(T)], 1), np.degrees(T)


def fit(model_fn, x0, bounds, b, target, name):
    res = least_squares(lambda p: (model_fn(b, p) - target).ravel(), x0,
                        bounds=bounds, method="trf", max_nfev=20000)
    err = np.linalg.norm(model_fn(b, res.x) - target, axis=1)
    return res.x, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov-deg", type=float, default=90.0)
    ap.add_argument("--json", default="SLAM/configs/fitted_camera_models.json")
    args = ap.parse_args()

    b, theta = sample_bearings(args.fov_deg)
    out = {}
    for cam, kb in KB4.items():
        target = kb4_project(b, kb)
        fx, fy, cx, cy = kb[:4]
        print(f"\n================ {cam}  (fit over 0–{args.fov_deg:.0f}°, {len(b)} rays)")
        out[cam] = {}

        # Mei: for small theta, u ~ fx_kb*theta and u ~ fx_mei*theta/(1+xi)
        xi0 = 1.0
        x0 = [xi0, fx * (1 + xi0), fy * (1 + xi0), cx, cy, 0.0, 0.0, 0.0, 0.0]
        lo = [0.0, 1.0, 1.0, cx - 50, cy - 50, -2, -2, -0.5, -0.5]
        hi = [5.0, 1e4, 1e4, cx + 50, cy + 50, 2, 2, 0.5, 0.5]
        p, err = fit(mei_project, x0, (lo, hi), b, target, "mei")
        out[cam]["mei"] = dict(params=list(map(float, p)),
                               names=["xi", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
                               rms_px=float(np.sqrt((err**2).mean())), max_px=float(err.max()))
        print(f"  Mei  rms={np.sqrt((err**2).mean()):.4f}px max={err.max():.4f}px  "
              f"xi={p[0]:.5f} fx={p[1]:.3f} k1={p[5]:.5f} k2={p[6]:.5f}")

        # EUCM
        x0 = [0.6, 0.9, fx, fy, cx, cy]
        lo = [0.0, 0.05, 1.0, 1.0, cx - 50, cy - 50]
        hi = [1.0, 5.0, 1e4, 1e4, cx + 50, cy + 50]
        p, err = fit(eucm_project, x0, (lo, hi), b, target, "eucm")
        out[cam]["eucm"] = dict(params=list(map(float, p)),
                                names=["alpha", "beta", "fx", "fy", "cx", "cy"],
                                rms_px=float(np.sqrt((err**2).mean())), max_px=float(err.max()))
        off = EUCM_OFFICIAL[cam]
        print(f"  EUCM rms={np.sqrt((err**2).mean()):.4f}px max={err.max():.4f}px  "
              f"alpha={p[0]:.6f} beta={p[1]:.6f} fx={p[2]:.4f}")
        print(f"       official     alpha={off[0]:.6f} beta={off[1]:.6f} fx={off[2]:.4f}"
              f"   -> dalpha={p[0]-off[0]:+.6f} dbeta={p[1]-off[1]:+.6f} dfx={p[2]-off[2]:+.4f}")

        # Double Sphere
        x0 = [0.3, 0.6, fx, fy, cx, cy]
        lo = [-1.0, 0.0, 1.0, 1.0, cx - 50, cy - 50]
        hi = [1.0, 1.0, 1e4, 1e4, cx + 50, cy + 50]
        p, err = fit(ds_project, x0, (lo, hi), b, target, "ds")
        out[cam]["ds"] = dict(params=list(map(float, p)),
                              names=["xi", "alpha", "fx", "fy", "cx", "cy"],
                              rms_px=float(np.sqrt((err**2).mean())), max_px=float(err.max()))
        print(f"  DS   rms={np.sqrt((err**2).mean()):.4f}px max={err.max():.4f}px  "
              f"xi={p[0]:.5f} alpha={p[1]:.5f} fx={p[2]:.3f}")

    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
