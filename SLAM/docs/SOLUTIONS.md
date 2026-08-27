# What the Hilti×Trimble 2026 Field Actually Did

Distilled from all 19 submitted reports (11 SLAM track, 8 localization track). Rank numbers below are **SLAM-track** unless marked `L`. Reference so we never have to re-read the PDFs.

---

## The Cross-Team Table

| layer | what most top teams do | outliers / notes |
|---|---|---|
| **Base** | Keyframe-BA dominates: OKVIS2-X (#1, #7), ORB-SLAM3 (#5, #11, #15). Filters: OpenVINS (#13, #14), √VINS (#2) | #2 is a **square-root** filter — numerically far better than plain MSCKF, and it placed 2nd |
| **Camera model** | EUCM most-named (#2, #7, #14); KB4 elsewhere | #15 stitched to equirect — placed last of the group, and its own conclusion says raw would have been better |
| **Input** | Raw fisheye (#1, #2, #7, #9, #13, #14) | #8 rendered 12 virtual perspective views — placed 8th |
| **Features** | Classical majority; **lines (ELSED) only in #1 and #2** | Lines are the clearest top-2-only signal in the whole field |
| **Rig** | cam0-anchored + camera-indexed residuals (#1, #2); stock multi-cam (#7, #13, #14) | #11 discarded cam1 entirely; **nobody attempts simultaneous stereo** |
| **Loops** | Cross-camera aware in all three top entries | #13 none, #14 built then rejected it, #15 panoramic |
| **IMU init** | **Dynamic init** — universal, repeatedly called the single biggest fix | ZUPT tried by 3 teams, all reported useless (continuous walking) |
| **Calibration** | Provided as-is (#8, #13, #14, #15) | Online extrinsics (#7); re-calibrated (#5); **L#2 found extrinsics drift between runs** (thermal + mount play) |
| **Rolling shutter** | **Nobody in the SLAM track** | Only the localization winner (L#1) — who won |
| **Masking** | Static self-occlusion masks are standard (#7 from organisers, #2) | #1 uses geometric height rejection instead — no semantics |
| **Global refinement** | Offline BA (#8, #13); non-causal explicitly allowed | #13 measured 0.327 → 0.262 m from BA alone |

**The winning recipe is not "copy #1" — it is the intersection:** raw fisheye · EUCM-family model · angular-uniform feature selection · lines · cam0-anchored rig · cross-camera two-stage loops · dynamic init. Then our additions, of which **rolling shutter is a genuine hole in the entire SLAM track**.

---

## Leaderboard → What It Costs

Totals are sums over 25 sequences. Score per sequence `S = 100·exp(−0.4605·e)`.

| rank | team | total | per-seq | ≈ error |
|---|---|---|---|---|
| 1 | ACDC-K (KAIST) | 2410.0 | 96.4 | 0.08 m |
| 2 | SPARO (Inha) | 2393.9 | 95.8 | 0.09 m |
| 3 | undisclosed | 2320.7 | 92.8 | 0.16 m |
| 5 | SnT-ARG (Luxembourg) | 2299.3 | 92.0 | 0.18 m |
| 7 | MRL (TUM/ETH) | 2292.6 | 91.7 | 0.19 m |
| 9 | Monado's Basalt (TUM CVG) | 2235.8 | 89.4 | 0.24 m |
| 10 | ETH-THU-Stuttgart | 2217.3 | 88.7 | 0.26 m |
| 15 | PanoAir (SYSU) | 2078.0 | 83.1 | 0.40 m |
| 25 | CNRS-AIST JRL | 1226.8 | 49.1 | 1.55 m |

Top-10 needs ~89 · top-5 needs ~92 · **winning needs ~96.4**. Note how brutal the top is: rank 1 → rank 7 is 0.08 m vs 0.19 m. Winning means roughly halving 7th place's error.

---

## Per-Team Detail

### #1 — ACDC-VSLAM, KAIST (2410.0)
Base **OKVIS2-X**. Names three coupled problems: (P1) wide-FOV distortion biases pixel-space feature selection, (P2) textureless indoor scenes, (P3) non-overlapping front/rear.
- **Spherical equal-area feature bucketing.** Back-project keypoints to bearings → spherical `(θ,φ)` → map to an equal-area 2D domain → quadtree bucket there. Each cell = equal *angular* area, not equal pixel area. **Selection only — no image resampling.**
- **Spherical line features.** ELSED segments, endpoints lifted to bearings, represented by the **normal of the associated great circle**. Filtered by angular length; tracked with gating on direction, normal alignment, length consistency.
- **Floor-aware landmark rejection** — gravity-aligned height filter kills geometrically impossible landmarks. Geometric, not semantic.
- **Local BA reweighting:** by landmark track length (favour long tracks) and by inverse depth (favour near, well-parallaxed points).
- **Adaptive rear-camera deactivation.** The rear camera *hurts* during forward motion — it only sees receding, short-lived, weakly-conditioned features. Deactivated at keyframe boundaries when motion aligns with the front axis and the front has enough features; reactivated on rotation or front starvation. **Detection + per-camera BoW keep running**, so rear frames stay available for loops.
- **Two-stage dual loop detection.** Stage 1 intra-camera (f→f, r→r); stage 2 inter-camera (f→r, r→f) only on failure. In this rig the dominant loop signal is cross-camera.
- **Adaptive manifold switching:** SE(3) for clean revisits · Sim(3) when scale drift is measured · **4-DoF (yaw+translation)** when correspondences are weak and IMU already pins roll/pitch · reject if even that fails.

### #2 — √VINS-360, Inha University + Riibotics (2393.9, mean ATE **0.109 m**)
Best raw accuracy in the field.
- Frontend **√VINS** (square-root filter), backend **GTSAM** factor graph. Non-causal.
- **SuperPoint + LightGlue on raw (conditioned) fisheye** — not rectified. State-predicted matching for fast motion.
- **ELSED lines** via MSCKF-style update (PL-VIWO). **Vertical-world / weak-Manhattan** assumption: gravity fixes vertical, horizontal left free.
- Conditioning: static masks · **Zero-DCE++** low-light enhancement · **YOLO26** to suppress moving objects.
- **One cam0 trajectory**; cam1 projects through a shared rigid `E_01`. No duplicate per-frame camera nodes.
- **Short-term cross-camera matching** in a bounded time window — front-now vs rear-a-moment-ago. Because these span the calibrated baseline they inject **local metric scale**.
- Loops via **SALAD** global descriptors + SuperPoint/LightGlue verification, inserted as EUCM-compatible **structureless smart projection factors**.
- **Global scale variable `s`** in the graph, tightly regularised so it absorbs calibration error but not drift.

### #7 — MRL, TU Munich / ETH (2292.6)
OKVIS2-X off the shelf, by its own authors. **EUCM with analytic Jacobians.** Online IMU-to-camera extrinsics. `medianBlur` k=3 for dark scenes. **Static self-occlusion masks** adapted from the organisers. Dense mapping disabled — verbatim: *"due to the lack of enough overlap between the two cameras."* Single config for all sequences. Ran **six times and reported the best** — acknowledges RANSAC nondeterminism.

### #9 — Monado's Basalt, TUM CVG (2235.8)
Basalt, CPU-only, ~25 ms/frame on 4 threads. **Raw fisheye, no undistortion.** Patch-based pyramidal KLT. **IMU-PARSAC** outlier rejection specifically for *moving operator and flickering lights*. Async loop-closure thread with Ceres PGO. ATE 0.162–1.250 m — locally excellent, globally drifting.

### #13 — ETH D-ITET (two-stage)
**OpenVINS MSCKF → offline COLMAP VI-BA.** Measured **0.327 → 0.262 m** from BA alone. No loop closure at all. Key finding: at full playback the CPU only hit 16–23 fps against a 30 Hz stream, shortening every track — **running at 0.5× playback fixed it**. Dynamic init with a 1.0 s window cut one sequence's init delay from ~59 s to 5.6 s. Names its own main limitation: KLT carries no descriptors, so a lost track returns with a new ID and breaks the observation chain; recommends ALIKED + LightGlue.

### #14 — Floorplan-aware, independent (2135.2, mean ATE 0.503 m)
OpenVINS "Hilti EUCM fork", config-level only. **Enabling the second camera took ATE from 262 m → 0.29 m (901×).** Dynamic init lifted the GT-subset score 330 → 427/500. **CLAHE was actively harmful** — prevented initialisation entirely on one sequence (0 poses vs 191k). Built a full loop-closure system with DINOv2 and **rejected it** (helped 3 of 5 GT sequences, hurt 2). Backward IMU dead-reckoning to fill the pre-init prefix, because missing coverage scores zero.

### #15 — PanoAir, Sun Yat-sen (2078.0)
**Stitched the two fisheyes into one ERP panorama** and ran single-camera 360 VI-SLAM on ORB-SLAM3 with ERP Jacobians. Hybrid hand-crafted + learned features. SE₂(3) exact preintegration; rotation-translation decoupled init. Ships panoramic loop closure. **Its own conclusion says the single-optical-centre approximation costs accuracy and raw dual-fisheye would be better.** Explicitly does not handle occlusion.

### #8 — OmniRecon, ETH (2246.3)
COLMAP/GLOMAP panorama SfM. Renders **12 virtual perspective views** (4 yaw × 3 pitch, 90° FOV) because *"learned features degrade under Kannala-Brandt distortion"* — then placed 8th, behind teams running learned features on raw fisheye. **1/θ incidence blending + Voronoi seam assignment** so overlap isn't double-counted. Replaced COLMAP's vocab-tree loop detector with **MegaLoc**, averaging the 12 per-view descriptors into one **heading-invariant frame descriptor**. Had to patch GLOMAP because *its BA silently re-optimises rig extrinsics even with intrinsics fixed*. Carrier masking via prompt-conditioned YOLOE ∪ a static template.

### Localization track, worth stealing from
- **L#1 (Yizhou Zang, ORB-SLAM3)** — the only entry anywhere that handles **rolling shutter**: per-keypoint row-time offset, gyro-integrated `ΔR(τ)` for the frontend, and a BA projection edge evaluated **from the original measurement at its readout time** (not the corrected keypoint). Also rotation-aware IMU preintegration using `J₁(ωΔt)`, `J₂(ωΔt)`.
- **L#2 (ETH CVG, OKVIS2-X)** — found the **provided extrinsics vary across sequences**, blamed on mechanical play and thermal expansion, and turned on online extrinsic calibration.
- **L#3 (MVP-SLAM, Luxembourg)** — map points **shared between both cameras**, re-observed by whichever one sees them when the rig rotates.

---

## Hard Facts About the Rig (from the official calibration)

- **EUCM is native; KB4 is a fit to it.** The calibration file states it outright.
- `cam_overlaps: []` — non-overlap is official, not inferred.
- Baseline **4.01 cm**, cam0→cam1 rotation **179.56°** (computed from their extrinsics).
- Resolution 1472×1440, `f ≈ 465` → the image circle reaches ~90° off-axis.
- IMU noise densities: accel `2.86e-3`, gyro `4.70e-4`. Time offset `−6.57 ms`.
- **cam0/cam1 hardware skew measured at 0.0 ms** — no inter-lens time offset exists.
- Raw calibration rosbags + AprilTag configs (`april_6x6`, `april_7x12`) ship with the dataset, so Kalibr can be re-run.

---

## Universal Failure Modes

1. **Underground / textureless (UG1, UG2) destroys everyone.** Rank 1 drops to 92–95 there; PanoAir collapses to 55–64. The leaderboard is decided here, not on the easy floors.
2. **Initialisation** — static init waits for a stationary→moving jerk that never comes because the operator is already walking.
3. **Coverage is a hard gate.** Missing poses score zero, not "skipped".
4. **Repetitive structure causes false loops** — white walls and 90°-rotated corridors produce high-inlier matches that aren't real revisits. This, not distortion, is the reported loop-closure failure.
5. **Nondeterminism is real and acknowledged** — #7 ran six times and took the best.
