# Results

Every row is one measured configuration on **Hilti floor_EG_2025-12-02_run_1**
unless stated. Score `S = 100·exp(−0.46052·e)`, SE(3) aligned (metric), which is
how the challenge scores. Failures are kept — a table of only wins is how a dead
end gets re-tried six weeks later.

**Noise floor: 0.79 score.** ORB-SLAM3 is nondeterministic (3 identical runs:
90.89 / 91.68 / 91.48, keyframes 555–581). Any difference below ~0.8 is not a
result.

---

## Headline

| # | method (stepwise) | ATE | SCORE |
|---|---|---|---|
| 1 | ORB-SLAM3 mono-inertial, cam0, KB4, Step-0 rig loaded → **+ IMU metric scale** | **0.117 m** | **95.42** |
| — | *rank 1 reference (per-seq, over their 25 seqs)* | ~0.08 m | *96.4* |

---

## Estimator bake-off

| # | method | ATE | SCORE | notes |
|---|---|---|---|---|
| 2 | ORB-SLAM3 mono-inertial, cam0 only | 0.205–0.224 | 90.89–91.68 | 3 runs; uses HALF the rig |
| 3 | OpenVINS MSCKF, 2 cams | 0.235 | 90.31 | deterministic; no loop closure, no global BA |
| 4 | OKVIS2-X, 2 cams, masked | 0.492 | 85.88 | pre-final-BA trajectory |
| 5 | OKVIS2, 2 cams, masked, 3 bugs fixed | 0.667 | 81.03 | |
| 6 | OKVIS2, 2 cams, stock | 0.707 | 79.87 | |

A **mono** system beating both 2-camera keyframe-BA systems is the standing
anomaly. It is why ORB-SLAM3 was chosen as the base.

---

## Step 0 — per-unit calibration from images + IMU alone (no GT)

| # | quantity | recovered | truth | error |
|---|---|---|---|---|
| 7 | inter-lens angle (optical axis) | 179.438° | 179.561° | **−0.123°** |
| 8 | …vs assuming a perfect 180° rig | 180.000° | 179.561° | +0.439° (3.6× worse) |
| 9 | rig baseline | 4.01 cm | 4.01 cm | ~1 % |
| 10 | metric scale (IMU) | 2.4144 | 2.4422 | −1.1 % |

Method: pick sparse frames → pair `front[i]↔front[j]`, `back[i]↔back[j]`, and
**`front[i]↔back[i+Δ]`** for Δ ∈ {3,6,12,24,48} → COLMAP fisheye SfM → robust
per-frame median → rig-constrained BA → IMU visual-inertial alignment for scale.

The `front[i]↔back[i+Δ]` trick is what makes the rig observable at all: walking
forward, the REAR camera at *t+Δ* sees what the FRONT camera saw at *t*. 178 of
925 cross-camera pairs verified geometrically.

---

## Ablations that FAILED (do not retry blindly)

| # | change | result | why |
|---|---|---|---|
| 11 | Unblock `ScaleRefinement()` (KF≤200 + 25–75 s gates) | 90.85–91.26, **no change** | gate was real but not load-bearing; scale freezes elsewhere |
| 12 | Masks, 78° (inherited) | 90.29 | inside noise; also cost OKVIS2 ~1 pt |
| 13 | Masks, true image circle 88° | 90.60 | inside noise |
| 14 | Full VI-BA after scale correction | 92.58 | **worse than scale alone (95.42)** |
| 15 | VI-BA, visual term only | 93.53 | **also worse** — structure is triangulated FROM the poses the BA then moves; reprojection improves while ATE drifts |
| 16 | OKVIS2 overlap fix (`backProject` returns) | 79.87 → 82.13 | real fix, small gain |
| 17 | OKVIS2-X loop closures OFF | 85.88 → 81.60 | loops HELP by 4.3; they were not the BA poison |
| 18 | `rig_configurator` average for rig extrinsics | 178.118° (−1.44°) | naive mean; worse than assuming 180°. Use robust median → BA |

---

## Bugs found in upstream code

| # | where | bug |
|---|---|---|
| 19 | OKVIS2 + OKVIS2-X `NCameraSystem.cpp:87,99` | `backProject()` return value discarded → false cross-camera overlap on a back-to-back rig |
| 20 | OKVIS2 + OKVIS2-X `CameraBase.hpp:97` | `isInImage()` accepts NaN → `mask_.at<uchar>(int(NaN))` → segfault on any masked fisheye |
| 21 | OKVIS2 `CameraBase.hpp:100` | doc says `0 == masked`; `isMasked()` returns the pixel value, so **nonzero == masked**. Doc is wrong |
| 22 | ORB-SLAM3 `ORBextractor.cc` | `operator()` accepts a mask argument and ignores it entirely |
| 23 | ORB-SLAM3 examples | build the MapPoint cloud but never save it |
| 24 | our `step0_imu_scale.py` | preintegrated in IMU frame while rotating by CAMERA rotation → `\|b_a\|`=9.24, `\|g\|`=1.17, negative scale |
| 25 | our `align.json` emitter | emitted the UP vector as `gravity_dir` → `\|b_a\|` = 18.9 (two gravities) in VI-BA |

---

## Reproduce the headline

```bash
# 1. SLAM
docker run ... mono_inertial_euroc ORBvoc.txt \
    SLAM/configs/orbslam3_hilti/hilti_mono_inertial.yaml \
    Data/Hilti/orb/<seq> SLAM/configs/orbslam3_hilti/timestamps.txt orb

# 2. metric scale (triangulate -> VI align -> apply)
python SLAM/pipeline/run/apply_imu_scale.py \
    --traj f_orb.txt --dataset Data/Hilti/ds/<seq> \
    --imucam SLAM/configs/hilti_fast/kalibr_imucam_chain.yaml \
    --out f_orb_scaled.txt

# 3. run folder + score
python SLAM/pipeline/viz/make_run_output.py --engine tum \
    --traj f_orb_scaled.txt --dataset Data/Hilti/ds/<seq> \
    --out SLAM/experiments/<name> --gt Data/Hilti/groundtruth/<seq>.txt \
    --config SLAM/configs/hilti_fast
```

---

## Caveats on the headline

- **One sequence.** The other two Hilti sequences are unmeasured with this build.
- **floor_EG is an easy sequence.** SOLUTIONS.md: the leaderboard is decided on
  the underground sequences, not the easy floors.
- **The challenge's 25 sequences are not ours.** 95.42 here does not convert to a
  rank.
- Single sample against a 0.79 noise floor; the headline needs 3 runs to be a
  claim rather than a reading.
