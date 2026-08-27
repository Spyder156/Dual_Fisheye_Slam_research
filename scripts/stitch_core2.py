#!/usr/bin/env python3
"""
Arm C core, step 2: BOTH lenses -> full 360 equirect.
 - orientation fix: 180 deg roll about optical axis (input was upside down)
 - cam2 placed via extrinsic recovered from FIORD model/ (median R_12 = R2 @ R1^T)
 - overlay both lenses; blend the overlap band (feather by angular distance to FoV edge)
 - honest validity mask; overlap map
Writes: combined equirect, per-lens equirects, overlap map, one viz figure.
"""
import os, re, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
OUT = f"{ROOT}/outputs/armC"; os.makedirs(OUT, exist_ok=True)
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"

CAM1 = dict(fx=998.71442825746783, fy=1006.4807034467711, cx=1612.0, cy=1617.0,
            k=[0.034358190499964164, -0.01780110435914365, -0.00079837513258086007, 0.00036177684527421565],
            w=3264, h=3264)
CAM2 = dict(fx=997.51132933461327, fy=1005.2394553958525, cx=1612.0, cy=1617.0,
            k=[0.036087615485948625, -0.02348048096992612, 0.0042046925204551316, -0.00090948827268945594],
            w=3264, h=3264)
FOV_DEG = 200.0
ROLL = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                 [np.sin(np.pi),  np.cos(np.pi), 0], [0, 0, 1]])  # 180 about Z


def qvec2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def recover_R12():
    """median relative rotation R_12 = R2 @ R1^T from FIORD model poses."""
    R1, R2 = {}, {}
    lines = [l for l in open(BTXT) if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        R = qvec2R(*map(float, p[1:5])); name = p[9]
        m = re.search(r'_(\d+)_fisheye(\d)', name)
        if not m: continue
        (R1 if m.group(2) == '1' else R2)[m.group(1)] = R
    rels = []
    for k in set(R1) & set(R2):
        rels.append(R2[k] @ R1[k].T)
    rels = np.array(rels)
    # average rotation via mean then re-orthonormalize (SVD)
    M = rels.mean(0)
    U, _, Vt = np.linalg.svd(M)
    R12 = U @ Vt
    if np.linalg.det(R12) < 0:
        U[:, -1] *= -1; R12 = U @ Vt
    # consistency: angle of each rel vs mean
    angs = []
    for R in rels:
        d = R @ R12.T
        angs.append(np.degrees(np.arccos(np.clip((np.trace(d)-1)/2, -1, 1))))
    return R12, np.array(angs), len(rels)


def kb_project(X, Y, Z, cam):
    r = np.sqrt(X*X + Y*Y)
    theta = np.arctan2(r, Z)
    k1, k2, k3, k4 = cam["k"]; t2 = theta*theta
    theta_d = theta*(1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
    scale = np.where(r > 1e-9, theta_d/np.where(r > 1e-9, r, 1), 0.0)
    u = cam["fx"]*X*scale + cam["cx"]
    v = cam["fy"]*Y*scale + cam["cy"]
    return u, v, theta


def equirect_dirs(W, H):
    lon = (np.arange(W)+0.5)/W*2*np.pi - np.pi
    lat = np.pi/2 - (np.arange(H)+0.5)/H*np.pi
    lon, lat = np.meshgrid(lon, lat)
    X = np.cos(lat)*np.sin(lon); Y = -np.sin(lat); Z = np.cos(lat)*np.cos(lon)
    return np.stack([X, Y, Z], -1)


def render(fisheye, cam, dirs, R):
    d = dirs @ R.T
    X, Y, Z = d[..., 0], d[..., 1], d[..., 2]
    u, v, theta = kb_project(X, Y, Z, cam)
    tmax = np.deg2rad(FOV_DEG/2)
    valid = (theta <= tmax) & (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
    eq = cv2.remap(fisheye, np.where(valid, u, -1).astype(np.float32),
                   np.where(valid, v, -1).astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    eq[~valid] = 0
    # feather weight: fade near FoV edge
    w = np.clip((tmax - theta) / np.deg2rad(15), 0, 1) * valid
    return eq, valid, w


def main():
    fid = "729"
    R12, angs, n = recover_R12()
    print(f"extrinsic R_12 from {n} frames: rotation-consistency median={np.median(angs):.2f} deg, p90={np.percentile(angs,90):.2f} deg")
    ax_deg = np.degrees(np.arccos(np.clip((np.trace(R12)-1)/2, -1, 1)))
    print(f"cam1->cam2 relative rotation angle = {ax_deg:.1f} deg (expect ~180 for back-to-back)")

    f1 = cv2.imread(f"{FIR}/cam1/" + [f for f in os.listdir(f'{FIR}/cam1') if f'_{fid}_' in f][0])
    f2 = cv2.imread(f"{FIR}/cam2/" + [f for f in os.listdir(f'{FIR}/cam2') if f'_{fid}_' in f][0])

    W, H = 2048, 1024
    dirs = equirect_dirs(W, H)
    eq1, v1, w1 = render(f1, CAM1, dirs, ROLL)              # cam1: just orientation
    eq2, v2, w2 = render(f2, CAM2, dirs, R12 @ ROLL)        # cam2: extrinsic + orientation

    # blend
    wsum = w1 + w2 + 1e-6
    comb = (eq1*w1[..., None] + eq2*w2[..., None]) / wsum[..., None]
    comb = np.clip(comb, 0, 255).astype(np.uint8)
    valid = (v1 | v2)
    comb[~valid] = 0
    overlap = (v1 & v2)

    cv2.imwrite(f"{OUT}/combined_{fid}.png", comb)
    cv2.imwrite(f"{OUT}/overlap_{fid}.png", (overlap*255).astype(np.uint8))

    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2, 2, figsize=(22, 12))
    ax[0, 0].imshow(rgb(eq1)); ax[0, 0].set_title(f"cam1 equirect (valid {v1.mean()*100:.0f}%)")
    ax[0, 1].imshow(rgb(eq2)); ax[0, 1].set_title(f"cam2 equirect via recovered extrinsic (valid {v2.mean()*100:.0f}%)")
    ax[1, 0].imshow(rgb(comb)); ax[1, 0].set_title(f"COMBINED 360 (blended)  valid {valid.mean()*100:.0f}%")
    ax[1, 1].imshow(overlap, cmap="magma"); ax[1, 1].set_title(f"overlap band (both lenses)  {overlap.mean()*100:.1f}%")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"Arm C — both lenses, orientation-fixed, frame {fid}", fontsize=15)
    fig.tight_layout(); fig.savefig(f"{OUT}/both_lens_{fid}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("WROTE:", f"{OUT}/both_lens_{fid}.png , combined_{fid}.png , overlap_{fid}.png")


if __name__ == "__main__":
    main()
