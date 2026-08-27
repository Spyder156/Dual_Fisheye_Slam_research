#!/usr/bin/env python3
"""
Arm C stitch renderer — HARD SEAM, no blending (ghosts stay visible).

Model-agnostic by design: give it per-lens params + extrinsic R12 and it renders
the full-360 equirect, choosing each overlap pixel from the lens with the SMALLER
theta (more central = more reliable). Later model hypotheses just pass different
params here. Also dumps the lens-label (seam) map.

Default run = baseline Kannala-Brandt (FIORD params) into
outputs/armC/model_tests/baseline_kb/.
"""
import os, re, argparse, numpy as np, cv2

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"

CAM1 = dict(fx=998.71442825746783, fy=1006.4807034467711, cx=1612.0, cy=1617.0,
            k=[0.034358190499964164, -0.01780110435914365, -0.00079837513258086007, 0.00036177684527421565],
            w=3264, h=3264)
CAM2 = dict(fx=997.51132933461327, fy=1005.2394553958525, cx=1612.0, cy=1617.0,
            k=[0.036087615485948625, -0.02348048096992612, 0.0042046925204551316, -0.00090948827268945594],
            w=3264, h=3264)
FOV_DEG = 200.0
ROLL = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                 [np.sin(np.pi),  np.cos(np.pi), 0], [0, 0, 1]])


def qvec2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def recover_R12():
    R1, R2 = {}, {}
    lines = [l for l in open(BTXT) if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        p = lines[i].split(); R = qvec2R(*map(float, p[1:5])); name = p[9]
        m = re.search(r'_(\d+)_fisheye(\d)', name)
        if not m: continue
        (R1 if m.group(2) == '1' else R2)[m.group(1)] = R
    rels = np.array([R2[k] @ R1[k].T for k in set(R1) & set(R2)])
    U, _, Vt = np.linalg.svd(rels.mean(0)); R12 = U @ Vt
    if np.linalg.det(R12) < 0:
        U[:, -1] *= -1; R12 = U @ Vt
    return R12


def kb_project(X, Y, Z, cam):
    r = np.sqrt(X*X + Y*Y); theta = np.arctan2(r, Z)
    k1, k2, k3, k4 = cam["k"]; t2 = theta*theta
    td = theta*(1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
    s = np.where(r > 1e-9, td/np.where(r > 1e-9, r, 1), 0.0)
    return cam["fx"]*X*s + cam["cx"], cam["fy"]*Y*s + cam["cy"], theta


def equirect_dirs(W, H):
    lon = (np.arange(W)+0.5)/W*2*np.pi - np.pi
    lat = np.pi/2 - (np.arange(H)+0.5)/H*np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat)*np.sin(lon), -np.sin(lat), np.cos(lat)*np.cos(lon)], -1)


def project_lens(fisheye, cam, dirs, R):
    d = dirs @ R.T
    u, v, theta = kb_project(d[..., 0], d[..., 1], d[..., 2], cam)
    tmax = np.deg2rad(FOV_DEG/2)
    valid = (theta <= tmax) & (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
    eq = cv2.remap(fisheye, np.where(valid, u, -1).astype(np.float32),
                   np.where(valid, v, -1).astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return eq, valid, theta


def stitch_hard(f1, f2, cam1, cam2, R12, W, H):
    dirs = equirect_dirs(W, H)
    eq1, v1, t1 = project_lens(f1, cam1, dirs, ROLL)
    eq2, v2, t2 = project_lens(f2, cam2, dirs, R12 @ ROLL)
    # hard choice: smaller theta wins where both valid; else whoever's valid
    use1 = v1 & (~v2 | (t1 <= t2))
    use2 = v2 & (~v1 | (t2 < t1))
    comb = np.zeros_like(eq1)
    comb[use1] = eq1[use1]; comb[use2] = eq2[use2]
    label = np.zeros((H, W), np.uint8)      # 0 none, 1 cam1, 2 cam2
    label[use1] = 1; label[use2] = 2
    overlap = (v1 & v2)
    return comb, label, overlap, (v1 | v2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", default="729")
    ap.add_argument("--W", type=int, default=5760)
    ap.add_argument("--name", default="baseline_kb")
    a = ap.parse_args()
    H = a.W // 2
    OUT = f"{ROOT}/outputs/armC/model_tests/{a.name}"; os.makedirs(OUT, exist_ok=True)

    f1 = cv2.imread(f"{FIR}/cam1/" + [f for f in os.listdir(f'{FIR}/cam1') if f'_{a.fid}_' in f][0])
    f2 = cv2.imread(f"{FIR}/cam2/" + [f for f in os.listdir(f'{FIR}/cam2') if f'_{a.fid}_' in f][0])
    R12 = recover_R12()
    comb, label, overlap, valid = stitch_hard(f1, f2, CAM1, CAM2, R12, a.W, H)

    cv2.imwrite(f"{OUT}/equirect_{a.fid}.png", comb)
    # seam line = boundary between label 1 and 2
    seam = np.zeros_like(label)
    seam[:, :-1] |= (label[:, :-1] != label[:, 1:]).astype(np.uint8)
    seam_vis = comb.copy(); seam_vis[seam > 0] = (0, 0, 255)
    cv2.imwrite(f"{OUT}/equirect_{a.fid}_seamline.png", seam_vis)
    cv2.imwrite(f"{OUT}/overlap_{a.fid}.png", (overlap*255).astype(np.uint8))
    print(f"WROTE {OUT}/equirect_{a.fid}.png  ({a.W}x{H}, hard seam, no blend)")
    print(f"      overlap band = {overlap.mean()*100:.1f}% of sphere")


if __name__ == "__main__":
    main()
