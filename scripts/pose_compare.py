#!/usr/bin/env python3
"""
Trajectory comparison A (MediaSDK equirect) vs B (FIORD fisheye reference),
using the shared camera IDs as exact correspondences.

B center for an id = mean of its fisheye1/fisheye2 camera centers (rig center).
A center for an id = the pano camera center.
Umeyama sim3 aligns A -> B; residual = trajectory deviation caused by stitching.

Reported scale-free (as % of B trajectory extent) AND with a rough metric
scale estimate (flagged as approximate). Writes figures only.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
OUT = f"{ROOT}/outputs/viz/pose"; os.makedirs(OUT, exist_ok=True)
A_IMG = f"{ROOT}/outputs/benchmark/armA_mediasdk_equirect/sparse/0/images.txt"
B_IMG = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"


def qvec2R(w, x, y, z):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def read_centers(path):
    """return dict id -> list of camera centers"""
    d = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        q = list(map(float, p[1:5])); t = np.array(list(map(float, p[5:8])))
        name = p[9]
        R = qvec2R(*q); C = -R.T @ t
        ids = ''.join(ch if ch.isdigit() else ' ' for ch in name).split()
        # id = the frame number; for A name '729.png' -> 729; for B '..._729_fisheye1' -> 729
        key = None
        if 'fisheye' in name:
            import re
            m = re.search(r'_(\d+)_fisheye', name); key = m.group(1) if m else None
        else:
            import re
            m = re.search(r'(\d+)\.png', name); key = m.group(1) if m else None
        if key is None:
            continue
        d.setdefault(key, []).append(C)
    return d


def umeyama(src, dst):
    """sim3 mapping src->dst. returns s,R,t and transformed src."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t, (s * (R @ src.T).T + t)


A = read_centers(A_IMG)
B = read_centers(B_IMG)
shared = sorted(set(A) & set(B), key=int)
src = np.array([A[k][0] for k in shared])                 # pano center
dst = np.array([np.mean(B[k], axis=0) for k in shared])   # rig center (mean of fisheyes)
print(f"shared ids: {len(shared)}")

s, R, t, src_al = umeyama(src, dst)
res = np.linalg.norm(src_al - dst, axis=1)                # residual in B units
ext = np.linalg.norm(dst.max(0) - dst.min(0))             # B trajectory diagonal
pct = res / ext * 100

# rough metric scale: lidar room diagonal / B-cloud diagonal (APPROX, flagged)
lidar_diag = np.linalg.norm([8.113, 7.554, 2.238])
# B trajectory diagonal in B units already = ext; but room != trajectory, so approximate
approx_m_per_unit = lidar_diag / (np.linalg.norm(dst.max(0)-dst.min(0)))
res_m_approx = res * approx_m_per_unit

print(f"Umeyama scale={s:.4f}")
print(f"ATE vs B ref (B units): mean={res.mean():.4f} median={np.median(res):.4f} rms={np.sqrt((res**2).mean()):.4f}")
print(f"ATE as % of trajectory extent: mean={pct.mean():.2f}% median={np.median(pct):.2f}% p90={np.percentile(pct,90):.2f}%")
print(f"ATE approx metric (ROUGH, room-diag scale): median={np.median(res_m_approx)*100:.1f}cm mean={res_m_approx.mean()*100:.1f}cm")

# ---- viz ----
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax[0].plot(dst[:, 0], dst[:, 1], '-o', ms=3, c='k', label='B fisheye ref (rig center)')
ax[0].plot(src_al[:, 0], src_al[:, 1], '-o', ms=3, c='r', alpha=0.7, label='A MediaSDK equirect (aligned)')
for i in range(len(shared)):
    ax[0].plot([dst[i, 0], src_al[i, 0]], [dst[i, 1], src_al[i, 1]], c='gray', lw=0.5)
ax[0].set_title(f'Trajectories aligned (X-Y)  n={len(shared)}'); ax[0].legend(); ax[0].set_aspect('equal', 'box')

ax[1].hist(pct, bins=30, color='steelblue', alpha=0.8)
ax[1].axvline(np.median(pct), c='r', label=f'median {np.median(pct):.2f}%')
ax[1].set_xlabel('per-camera deviation A vs B (% of trajectory extent)')
ax[1].set_ylabel('count'); ax[1].legend()
ax[1].set_title('Stitching-induced trajectory deviation')
fig.tight_layout(); fig.savefig(f"{OUT}/pose_compare.png", dpi=120); plt.close(fig)

# per-id residual sorted
order = np.argsort(-res)
plt.figure(figsize=(14, 5))
plt.bar(range(len(shared)), pct[order], color='indianred')
plt.xlabel('camera (sorted worst->best)'); plt.ylabel('deviation (% traj extent)')
plt.title('Per-camera A-vs-B deviation'); plt.tight_layout()
plt.savefig(f"{OUT}/pose_per_camera.png", dpi=110); plt.close()

with open(f"{OUT}/pose_summary.txt", "w") as f:
    f.write(f"shared_ids={len(shared)} umeyama_scale={s:.4f}\n")
    f.write(f"ATE_Bunits mean={res.mean():.4f} median={np.median(res):.4f} rms={np.sqrt((res**2).mean()):.4f}\n")
    f.write(f"ATE_pct_extent mean={pct.mean():.2f} median={np.median(pct):.2f} p90={np.percentile(pct,90):.2f}\n")
    f.write(f"ATE_metric_APPROX median_cm={np.median(res_m_approx)*100:.1f} mean_cm={res_m_approx.mean()*100:.1f}\n")
print("WROTE:", OUT)
