#!/usr/bin/env python3
"""
Arm C backend fitter — overlap ray-consistency from manual correspondences.

Residual for a pair (p1,p2): unproject each pixel to a bearing ray, bring cam2's
ray into the rig frame via R12, measure the angle between them. Perfect model +
far point -> 0. We report the BASELINE (KB + COLMAP extrinsic) disagreement, then
fit hypotheses and compare. First test here: baseline vs extrinsic-refined.

Model-agnostic: swap `unproject` for DS/EUCM/KB+terms later; objective unchanged.

Outputs -> outputs/armC/model_tests/_fit/<tag>.png + prints residuals (deg & eq-px).
"""
import os, re, json, argparse, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
PAIRS = f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
BASE_EQ = f"{ROOT}/outputs/armC/model_tests/baseline_kb/equirect_729.png"
OUT = f"{ROOT}/outputs/armC/model_tests/_fit"; os.makedirs(OUT, exist_ok=True)
ROLL = Rot.from_euler('z', 180, degrees=True).as_matrix()
EQW = 5760


def qR(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def recover_R12():
    R1, R2 = {}, {}
    L = [l for l in open(BTXT) if not l.startswith("#") and l.strip()]
    for i in range(0, len(L), 2):
        p = L[i].split(); R = qR(*map(float, p[1:5])); m = re.search(r'_(\d+)_fisheye(\d)', p[9])
        if m: (R1 if m.group(2) == '1' else R2)[m.group(1)] = R
    rels = np.array([R2[k] @ R1[k].T for k in set(R1) & set(R2)])
    U, _, Vt = np.linalg.svd(rels.mean(0)); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R


def kb_unproject(u, v, cam):
    """pixel -> unit bearing ray in camera frame (Kannala-Brandt inverse)."""
    xp = (u - cam["cx"]) / cam["fx"]; yp = (v - cam["cy"]) / cam["fy"]
    thd = np.hypot(xp, yp)
    th = thd.copy()
    k1, k2, k3, k4 = cam["k"]
    for _ in range(12):                      # Newton solve theta_d(theta)=thd
        t2 = th*th
        f = th*(1+k1*t2+k2*t2**2+k3*t2**3+k4*t2**4) - thd
        df = 1+3*k1*t2+5*k2*t2**2+7*k3*t2**3+9*k4*t2**4
        th = th - f/df
    phi = np.arctan2(yp, xp)
    return np.stack([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)], -1)


def rays(pairs, cam1, cam2):
    r1 = kb_unproject(pairs[:, 0], pairs[:, 1], cam1)
    r2 = kb_unproject(pairs[:, 2], pairs[:, 3], cam2)
    return r1, r2


def angles(r1, r2, R12):
    r2r = r2 @ R12                            # R12^T r2  (row-vec convention)
    dot = np.clip((r1*r2r).sum(1), -1, 1)
    return np.degrees(np.arccos(dot))


def fit_extrinsic(r1, r2, R12_0):
    def res(x):
        R12 = Rot.from_rotvec(x).as_matrix() @ R12_0
        r2r = r2 @ R12
        return (r1 - r2r).ravel()
    s = least_squares(res, np.zeros(3), method='lm')
    return Rot.from_rotvec(s.x).as_matrix() @ R12_0


def eq_px(ray, W, H):
    d = ray @ ROLL.T                          # rig -> equirect(display) frame
    X, Y, Z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(X, Z); lat = np.arcsin(np.clip(-Y, -1, 1))
    return (lon+np.pi)/(2*np.pi)*W, (np.pi/2-lat)/np.pi*H


def draw_pairs(ax, eq, r1, r2, R12, title):
    H, W = eq.shape[:2]
    ax.imshow(cv2.cvtColor(eq, cv2.COLOR_BGR2RGB))
    r2r = r2 @ R12
    x1, y1 = eq_px(r1, W, H); x2, y2 = eq_px(r2r, W, H)
    for i in range(len(r1)):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], '-', c='yellow', lw=1.2)
        ax.scatter([x1[i]], [y1[i]], c='red', s=30, zorder=3)
        ax.scatter([x2[i]], [y2[i]], c='cyan', s=30, zorder=3)
    ax.set_title(title); ax.axis('off')


def main():
    d = json.load(open(PAIRS))
    cam1, cam2 = d["cam1"], d["cam2"]
    P = np.array([[p["u1"], p["v1"], p["u2"], p["v2"]] for p in d["pairs"]])
    r1, r2 = rays(P, cam1, cam2)
    R0 = recover_R12()

    a0 = angles(r1, r2, R0)
    R1 = fit_extrinsic(r1, r2, R0)
    a1 = angles(r1, r2, R1)

    # leave-one-out on the extrinsic refine
    loo = []
    for i in range(len(P)):
        m = np.ones(len(P), bool); m[i] = False
        Ri = fit_extrinsic(r1[m], r2[m], R0)
        loo.append(angles(r1[i:i+1], r2[i:i+1], Ri)[0])
    loo = np.array(loo)

    px = EQW/360.0
    print(f"pairs: {len(P)}")
    print(f"BASELINE  (KB + COLMAP extrinsic): median={np.median(a0):.3f}deg  mean={a0.mean():.3f}  max={a0.max():.3f}  | eq-px median={np.median(a0)*px:.1f}")
    print(f"EXTRINSIC-REFINED               : median={np.median(a1):.3f}deg  mean={a1.mean():.3f}  max={a1.max():.3f}  | eq-px median={np.median(a1)*px:.1f}")
    print(f"  leave-one-out (extrinsic)     : mean={loo.mean():.3f}deg  median={np.median(loo):.3f} (overfit guard)")

    eq = cv2.imread(BASE_EQ)
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    draw_pairs(ax[0], eq, r1, r2, R0, f"BASELINE  median disagreement {np.median(a0):.2f} deg  ({np.median(a0)*px:.0f} eq-px)")
    draw_pairs(ax[1], eq, r1, r2, R1, f"EXTRINSIC-REFINED  median {np.median(a1):.2f} deg  ({np.median(a1)*px:.0f} eq-px)   [red=cam1 ray, cyan=cam2 ray, line=residual]")
    fig.tight_layout(); fig.savefig(f"{OUT}/baseline_vs_extrinsic.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(P))
    plt.bar(x-0.2, a0, 0.4, label=f'baseline (med {np.median(a0):.2f} deg)', color='indianred')
    plt.bar(x+0.2, a1, 0.4, label=f'extrinsic-refined (med {np.median(a1):.2f} deg)', color='steelblue')
    plt.xlabel('correspondence #'); plt.ylabel('ray disagreement (deg)'); plt.legend()
    plt.title('Per-pair overlap disagreement'); plt.tight_layout()
    plt.savefig(f"{OUT}/per_pair.png", dpi=110); plt.close()
    print(f"WROTE {OUT}/baseline_vs_extrinsic.png , per_pair.png")


if __name__ == "__main__":
    main()
