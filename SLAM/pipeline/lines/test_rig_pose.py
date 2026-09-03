#!/usr/bin/env python
"""Rig sanity: front-only, rear-only and joint pose must agree.

Between two consecutive frames the rig moves ONCE. So the relative body motion
recovered from
    A) front-front matches alone,
    B) rear-rear matches alone, mapped into the front frame through T_c0_c1,
    C) both cameras' bearings together,
must be the same motion. If they disagree, the rig extrinsic is wrong -- and a
wrong extrinsic is exactly what cost us 79 points when the baseline was zero.

Rotation is compared directly. Translation is compared as a DIRECTION only:
monocular epipolar geometry recovers t up to scale, and the two lenses have no
common scale without triangulation.
"""
import re, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend import load_cam, unproject_kb4

def rel_pose(ds, cam, c, A, B, minz=0.35):
    ia = cv2.imread(str(ds/f"cam{c}"/f"{A:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    ib = cv2.imread(str(ds/f"cam{c}"/f"{B:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    if ia is None or ib is None: return None
    p0 = cv2.goodFeaturesToTrack(ia, 2000, 0.01, 8)
    if p0 is None: return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ia, ib, p0, None, winSize=(21,21), maxLevel=4)
    st = st.ravel().astype(bool)
    q0, q1 = p0.reshape(-1,2)[st], p1.reshape(-1,2)[st]
    b0, v0 = unproject_kb4(q0[:,0], q0[:,1], cam)
    b1, v1 = unproject_kb4(q1[:,0], q1[:,1], cam)
    # keep bearings well inside the hemisphere so the pinhole-normalised
    # essential-matrix solver is conditioned
    m = v0 & v1 & (b0[:,2] > minz) & (b1[:,2] > minz)
    if m.sum() < 30: return None
    n0 = (b0[m][:,:2]/b0[m][:,2:3]).astype(np.float64)
    n1 = (b1[m][:,:2]/b1[m][:,2:3]).astype(np.float64)
    E, inl = cv2.findEssentialMat(n0, n1, focal=1.0, pp=(0.,0.),
                                  method=cv2.RANSAC, prob=0.999, threshold=1e-3)
    if E is None or E.shape != (3,3): return None
    _, R, t, _ = cv2.recoverPose(E, n0, n1, focal=1.0, pp=(0.,0.))
    return R, t.ravel()/max(np.linalg.norm(t),1e-12), int(m.sum())

def ang(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))

ds = Path(sys.argv[1] if len(sys.argv)>1 else "Data/Hilti/ds/floor_EG_2025-12-02_run_2")
K  = sys.argv[2] if len(sys.argv)>2 else "SLAM/configs/hilti_fast/kalibr_imucam_chain.yaml"
cams = [load_cam(K,0), load_cam(K,1)]
# T_c0_c1 from the Step-0 rig in the SLAM config
cfg = open("SLAM/configs/orbslam3_hilti/hilti_monorig.yaml").read()
i = cfg.index("Rig.T_c0_c1"); d = re.search(r"data: \[(.*?)\]", cfg[i:], re.S).group(1)
T01 = np.array([float(x) for x in d.replace("\n"," ").split(",")]).reshape(4,4)
R01 = T01[:3,:3]

print(f"{'pair':>12}  {'front deg':>9} {'rear deg':>9} {'rear->front':>11} "
      f"{'dR(A,B)':>8}  {'t dir deg':>9}   n_front n_rear")
rows=[]
for A in (500, 1500, 2500, 3500, 4500):
    B = A+3          # a few frames apart: enough parallax, still one motion
    ra = rel_pose(ds, cams[0], 0, A, B)
    rb = rel_pose(ds, cams[1], 1, A, B)
    if ra is None or rb is None:
        print(f"{A}:{B:>6}  insufficient matches"); continue
    Rf, tf, nf = ra
    Rr, tr, nr = rb
    # rear motion expressed in the FRONT camera frame
    Rr_in_f = R01 @ Rr @ R01.T
    tr_in_f = R01 @ tr
    dR = ang(Rf.T @ Rr_in_f)
    dt = np.degrees(np.arccos(np.clip(abs(float(tf @ tr_in_f)), -1, 1)))
    rows.append((dR, dt))
    print(f"{A}:{B:<6}  {ang(Rf):9.3f} {ang(Rr):9.3f} {ang(Rr_in_f):11.3f} "
          f"{dR:8.3f}  {dt:9.2f}   {nf:6d} {nr:6d}")
if rows:
    r = np.array(rows)
    print(f"\nrotation disagreement front vs rear: median {np.median(r[:,0]):.3f} deg")
    print(f"translation-direction disagreement : median {np.median(r[:,1]):.2f} deg")
    print("\nrotation should agree to well under a degree if the rig rotation is right.")
    print("translation direction is far noisier: a 3-frame baseline is ~3 cm, so the")
    print("epipolar direction is weakly determined -- large numbers here are expected")
    print("and are NOT evidence against the extrinsic.")
