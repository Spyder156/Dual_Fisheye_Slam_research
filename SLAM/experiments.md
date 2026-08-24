# Dual-Fisheye SLAM — Experiment Tracker

Checklist of things to build, try, and vary. **Results and discussion go in chat; visualizations go in `SLAM/results/`.** This file only tracks *what to try* and its status.

Status key: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (reason noted)

**Benchmark target:** Hilti-Trimble SLAM track. Rank 1 (ACDC-VSLAM, KAIST, OKVIS2-X) = score 2410. Rank 2 (√VINS-360, Inha) = score 2394, **mean ATE RMSE 0.109 m**. Everyone else 0.25–0.5 m. Universal worst case = underground/textureless sequences.

---

## The Plan

**Goal:** beat 2410 on the Hilti×Trimble SLAM track. See `SOLUTIONS.md` for what the field did.
**Bar:** ~89/seq for top-10 · ~92 for top-5 · **~96.4 to win**. Decided on the underground sequences, not the easy floors.

### Phase 0 — Measurement you can trust *(mostly done)*
Deterministic runs · fixed metric set · GT scoring · one committed config per experiment. Without this every A/B is noise.

### Phase 1 — Baseline *(done)*
Stock OpenVINS + our fixes on floor_EG. **83.5 mean score** (0.307 / 0.154 / 0.877 m). This is the number to beat, not the thing we ship.

### Phase 2 — Calibration *(in progress)*
COLMAP-refined camera models, bake-off across KB4 / Mei / EUCM / Double Sphere, per-model params fed back into the estimator.

### Phase 3 — Move to the real base
**OKVIS2.** Keyframe-BA + loop closure. The base decides the ceiling: OpenVINS teams landed 13th–14th, OKVIS2 teams landed 1st and 7th. OKVIS2 ships KB4 natively and the challenge provides KB4 intrinsics, so this needs **no new camera model to start**.

### Phase 4 — The intersection recipe (what the winners share)
Raw fisheye · EUCM-family model · **equal-area angular feature selection** · **lines** · cam0-anchored rig · **cross-camera two-stage loops** · dynamic init.

### Phase 5 — Our additions (where we pass them)
1. **Rolling shutter** — a genuine hole in the entire SLAM track.
2. **Rig-constrained global BA** — already working for us at 0.99 px.
3. **Metric scale anchored on the 4 cm baseline**, not the accelerometer.
4. **Zero-parallax operator detection** — geometric, no semantic model.
5. **Rotation-first estimation** — the research bet.

### Locked Architecture

| layer | decision |
|---|---|
| **Base** | OKVIS2 (OKVIS2-X has no public release; its extras are LiDAR/GNSS/depth we don't have) |
| **Camera model** | Decided empirically — COLMAP bake-off (§1) |
| **Input** | Raw fisheye. No perspective rendering, no equirectangular. Ever. |
| **Features** | Classical + lines; fast-learned alternatives benchmarked (§2) |
| **Rig** | cam0-anchored + camera-indexed residuals |
| **Loops** | Cross-camera aware, two-stage, Sim(3) default |
| **IMU init** | Dynamic |
| **Calibration** | COLMAP refinement pass, masks tested |
| **Rolling shutter** | Experimental (§6) |
| **Masking** | Geometric height rejection primary; semantic masks tested in the COLMAP step |
| **Refinement** | Sliding-window VI-BA, scale anchored on the rig baseline (§9) |

---

## Lessons Already Paid For — do not repeat

- `num_opencv_threads` is parsed by OpenVINS but **`cv::setNumThreads()` is never called** by the library. Each app must apply it.
- **Nondeterminism was `init_dyn_mle_max_threads`** (multi-threaded Ceres in the initializer), not OpenCV threads and not ASLR. A FP summation difference produced 326 m vs 1804 m trajectories — the filter is *chaotic* on this data.
- **`init_max_features`, not `num_pts`,** governs the tracker until initialisation completes. Split across cameras.
- **`Grider_GRID` masked *after* truncating to top-N by response** — the strongest corners sit on already-tracked features, so cells returned nothing. Filter first, then take top-N.
- **ROS2 CDR alignment is relative to the message body** (after the 4-byte encapsulation header), and `float64` needs 8-byte alignment. Getting this wrong produced 1e308 IMU values.
- **COLMAP `ba_refine_principal_point` defaults to 0.** And without `--ImageReader.camera_params` you are measuring COLMAP's initial guess, not a refinement.
- **Score by interpolating to GT timestamps**, not nearest-neighbour — GT sits on a different phase grid and NN matching throws away 60% of the trajectory.
- **Never log `static=True` to Rerun** — it hangs viewer 0.33 on images.
- Always **gravity-gate** a freshly parsed IMU stream (mean |a| ∈ 8.5–11.0) before writing a dataset.

## 0. Infrastructure — do these first, everything else is unmeasurable without them

- [x] **Deterministic mode — SOLVED 2026-08-23.** Root cause was **`init_dyn_mle_max_threads: 6`** — the dynamic initializer's Ceres solve is multi-threaded, and multi-threaded Ceres is nondeterministic in FP summation order. Set to **1** → byte-identical runs.
  - Also fixed en route: OpenVINS parses `num_opencv_threads` but **never calls `cv::setNumThreads()`** — every app must apply it itself, and our runner didn't. The setting was inert for every run we had ever done. Now applied (0 = threading off).
  - Ruled out by experiment: OpenCV threading (spread persisted with it fully disabled), ASLR / pointer-keyed `unordered_map` iteration (spread persisted with ASLR off).
  - **Implication beyond determinism:** a floating-point summation difference in the initializer amplified into 326 m vs 1804 m trajectories. The filter is not merely noisy on this data — it is **chaotic**, i.e. operating at the edge of stability. Strong argument for a keyframe-BA system that re-linearizes over a filter that commits.
- [ ] Run each config **3×**, report spread not a single number.
- [ ] Fixed metric set for every run: ATE RMSE, per-turn yaw error vs gyro, loop-closure gap, coverage %, features-used/frame, reprojection RMS.
- [ ] Auto-generate the standard Rerun recording for every run (already scripted).
- [ ] One committed config file per experiment. No scratchpad-only configs (we lost a good one that way).
- [ ] Get the Hilti-Trimble dataset (needs approval — multi-GB).
- [ ] Reproduce a published baseline on it before trusting any of our numbers.

## 1. Camera Model & Calibration

- [ ] Fit **EUCM** and **Double Sphere** numerically to our validated factory Mei (sample bearings → project → least squares). Measure px residual. If < 0.2 px we lose nothing by switching.
- [ ] Add **Mei / UCM / EUCM / Double Sphere** to COLMAP's model registry (`src/colmap/sensor/models.h`) — COLMAP ships none of them.
- [ ] Model bake-off via COLMAP: masked frames, same images, each model, compare converged reprojection error + registered-image count + track length.
  - [ ] Guard against overfitting: more params always fits better. Compare on **held-out images**, or use BIC/AIC.
- [ ] Measure the **actual per-lens FOV in the 2944×2880 video crop**. We still don't know it, and it bounds everything.
- [ ] Re-solve GT hand-eye extrinsic **per sequence** — check for the thermal/mechanical drift that Hilti rank 2 reported.
- [ ] Online extrinsic refinement: off / on / tightly-regularised.
- [ ] Intrinsics in BA: frozen / free / with priors.

## 2. Frontend — Detection & Description

- [ ] **Equal-area spherical bucketing vs plain pixel-grid bucketing.** Isolate this one. Expected to be large — pixel-grid bucketing packs features into the image centre and starves the periphery.
  - [ ] Vary cell count and quadtree depth.
  - [ ] Works with any detector — test with ORB *and* SuperPoint.
- [ ] Detector bake-off, ordered by cost:
  - Classical: **FAST+ORB**, SIFT
  - Fast learned: **XFeat**, **ALIKED**, EdgePoint2, SiLK, GCNv2
  - **ZippyPoint** — binary descriptors, designed as an ORB-SLAM drop-in. Highest value/effort ratio if it works.
  - Heavy learned (keyframes/loops only): SuperPoint+LightGlue, RoMa, LoMa
- [ ] Matcher pairing matters as much as detector: brute-force vs LightGlue vs learned-dense. Benchmark detector×matcher, not detector alone.
- [ ] Feature budget: 250 / 500 / 800 / 1500.
- [ ] **Line features (ELSED).** Points-only vs points+lines. *Both top-2 teams use lines; nobody else does.* Prime suspect for the textureless failure.
  - [ ] Spherical line representation (great-circle normals) vs naive 2D lines.
  - [ ] Vary min angular length, tracking gates (direction, normal alignment, length consistency).
- [ ] Low-light enhancement: none / CLAHE / Zero-DCE++. **Note:** one team found CLAHE *prevented initialisation entirely* on a dark sequence (0 poses vs 191k). Verify on ours before trusting it.
- [ ] Image denoise: none vs `medianBlur` k=3 (used by the OKVIS2-X team for dark scenes).
- [ ] FAST threshold sweep: 5 / 10 / 15 / 20. (On Hilti frames, th=15 gives ~1600–2500 raw corners; th=5 gives ~9000.)
- [ ] Detection cadence: every frame vs keyframes only.
- [ ] Non-max suppression radius / `min_px_dist`: 8 / 15 / 25.
- [ ] Response type: FAST score vs Harris ordering inside a bucket.
- [ ] Pyramid levels for detection and for KLT (independent knobs).
- [ ] Sub-pixel refinement on/off, window size.
- [ ] Per-camera feature budget split: equal vs weighted toward whichever camera is better conditioned.

## 3. Frontend — Tracking & Matching

- [ ] KLT vs descriptor matching vs LightGlue.
- [ ] **Hybrid schedule:** cheap tracking every frame, neural detect+match at keyframes only (2–5 Hz). Measure accuracy vs cost against all-neural.
- [ ] **Gyro-predicted search windows** on/off. Our gyro is certified to 0.128° — this should help most exactly where tracking fails (fast turns).
- [ ] Re-derive the RANSAC threshold **in angular terms**. Ours is currently a hacked 15 px.
- [ ] Angular vs pixel outlier gates throughout.
- [ ] Track-length reweighting on/off.
- [ ] Inverse-depth reweighting on/off (down-weight distant, low-parallax points).
- [ ] Playback rate: full speed vs 0.5×. One team found the tracker silently degraded when the CPU couldn't keep up — 16–23 fps against a 30 Hz stream shortened every track. Check we're not doing this.
- [ ] KLT window size 15/21/31 and pyramid depth 3/4/5.
- [ ] Forward-backward consistency check on/off, threshold sweep.
- [ ] Max track length / forced re-detection interval.
- [ ] Track re-identification across breaks (descriptor re-association) — #13's named limitation.
- [ ] Match search radius from gyro-predicted position vs fixed radius.

## 4. Rig / Multi-Camera

- [ ] Config sweep: mono cam0 / mono cam1 / dual always-on / dual with adaptive rear.
- [ ] **cam0-anchored parameterisation**: one pose per timestamp labelled as cam0; cam1 observations projected through fixed `E_01`. No duplicate camera nodes.
- [ ] **Adaptive rear deactivation** (rank 1's finding — the rear *hurts* during forward motion because it only sees receding, short-lived features).
  - [ ] Vary the trigger: motion-vs-front-axis angle, front feature count.
  - [ ] Confirm rear detection + descriptor DB registration keep running while rear is excluded from BA.
- [ ] **Short-term cross-camera temporal matching** (rank 2): bounded time window, front-now vs rear-a-moment-ago. On/off; vary window length.
- [ ] Test the inverse hypothesis: **weight cam1 higher during high-gyro moments.** Nobody does this. During a turn the front sweeps features out while the rear sweeps them in.
- [ ] Baseline (4 cm) as hard metric constraint vs soft prior vs unused.
- [ ] Measure how often cam1 actually contributes a constraint that cam0 could not. This tells us what the rig is really worth.
- [ ] Extrinsic perturbation study: inject 0.5°/1°/2° and 1/5/10 mm errors, measure score sensitivity. Tells us how much calibration accuracy is actually worth.
- [ ] Baseline scale sweep: 3.9 / 4.01 / 4.1 cm — how sensitive is metric scale?
- [ ] Per-sequence vs global extrinsics (L#2 found drift between runs).
- [ ] Fraction of accepted loops that are cross-camera — the number that says what the rig is worth.

## 5. IMU & Initialisation

- [ ] Static vs **dynamic init** — every team that reported on this called it their single biggest fix. Static init waits for a stationary→moving jerk that never comes when the operator is already walking.
- [ ] Init window: 0.5 / 1.0 / 2.0 / 3.0 s.
- [ ] Init on front only vs both cameras.
- [ ] IMU noise densities: measured (from the static/6-face recording) vs generic vs inflated.
- [ ] Preintegration: Forster on-manifold vs rotation-aware closed-form vs SE₂(3) exact.
- [ ] Rotation-translation **decoupled** initialisation.
- [ ] ZUPT on/off. Expect useless — three teams independently reported no stationary segments in continuous walking.
- [ ] Gyro-only rotation prior vs full IMU coupling.
- [ ] **Accelerometer health gate:** our accel failed its test (integrating it was *worse* than assuming constant velocity, 25 cm vs 14 cm per 1 s window). Re-test after the static + 6-face recording before trusting it for scale.
- [ ] Use the official Hilti noise densities (accel 2.86e-3, gyro 4.70e-4) vs inflated vs our guesses.
- [ ] Noise scaling sweep ×0.5 / ×1 / ×2 / ×5 around the official values.
- [ ] Gravity magnitude: 9.81 vs locally correct value.
- [ ] Initial bias: zero vs estimated from the first stationary window.
- [ ] Backward IMU dead-reckoning to fill the pre-init prefix (#14's coverage trick) — coverage is a hard gate.
- [ ] Init disparity threshold sweep; measure init latency per sequence.

## 6. Rolling Shutter — *zero SLAM-track teams do this*

- [ ] Readout sweep on a GT dataset: 0 / ±8 / ±16 / ±25 ms. Confirm our measured −16 ms.
- [ ] Frontend row-time correction only / BA row-time edge only / both.
  - Note: the BA edge should evaluate the residual from the **original measurement at its readout time**, not from the frontend-corrected keypoint.
- [ ] Gyro-only row correction vs gyro + translation.
- [ ] Verify RS error correlates with turn rate. If it doesn't, our −16 ms is fitting something else.

## 7. Back-End

- [ ] **Angular residuals vs pixel residuals.** Isolate. On a 200° lens, pixel error silently over-weights the centre by ~10× versus the periphery.
- [ ] Robust kernel: none / Huber / Cauchy; sweep thresholds.
- [ ] Structureless smart factors vs explicit landmarks.
- [ ] Global scale variable in the graph: on/off, vary prior tightness.
- [ ] Sliding-window size / clone count sweep.
- [ ] Marginalisation strategy.
- [ ] Gravity-aligned **height rejection** of geometrically impossible landmarks (rank 1's floor-aware filter). Geometric, no semantics.
- [ ] chi2 multiplier sweep for MSCKF and SLAM updates.
- [ ] Number of SLAM landmarks kept in state: 25 / 50 / 100.
- [ ] Feature marginalisation policy: max track length before forced use.
- [ ] FEJ on/off.
- [ ] Update decimation: track at 30 Hz, update at 10/15/30 Hz.

## 8. Loop Closure

- [ ] **Two separate descriptor databases, one per camera.** Four query paths: f→f, r→r, f→r, r→f.
- [ ] **Two-stage retrieval**: intra-camera first (cheap, same-direction revisits), inter-camera only on failure (catches reverse-direction revisits that same-camera search is structurally blind to).
- [ ] Candidates carry camera-pair provenance so verification uses the right projection models.
- [ ] Descriptor bake-off: **SALAD** (rank 2's choice) vs **BoQ** vs DINOv2. *(NetVLAD/MegaLoc excluded per preference.)*
- [ ] Manifold: **Sim(3) as default** (scale drift is endemic on fisheye) vs SE(3) vs 4-DoF fallback vs adaptive switching.
- [ ] Rejection budget for weak loops — one team measured loop closure helping 3 of 5 sequences and *hurting* 2. Gate hard.
- [ ] **Measure what fraction of accepted loops are cross-camera.** This is the number that says whether the rig earns its keep.
- [ ] False-loop rejection on repetitive structure (white walls, identical corridors) — the dominant reported failure, not distortion.
- [ ] Verification thresholds: min inliers, min score, temporal gap.
- [ ] Loop acceptance rate and false-positive audit against GT (we can label true revisits from GT positions).
- [ ] Descriptor on raw fisheye vs on a gravity-aligned crop.
- [ ] Effect of loop closure on the *hard* sequences specifically, not the aggregate.

## 9. Online Sliding-Window VI-BA + Global Refinement

**The scale question (raised 2026-08-23, and it's the right question).** Concern: an online VI-BA's visual scale fights the IMU's metric scale. In a *tightly-coupled* graph there is no fight by construction — IMU and visual factors sit in one optimization and scale is jointly observable. The fight is real in three cases: (a) visual BA run separately then merged, (b) low IMU excitation, (c) **a bad accelerometer — which is our case** (ours integrated *worse* than assuming constant velocity). So:

- [ ] **Anchor scale on the 4 cm rig baseline, not the accelerometer.** Hard geometric constant, always available, already proven (rig-constrained COLMAP recovered 0.58 m/s correctly). Accel becomes a secondary source, gated on the IMU health test.
- [ ] Note: overlapping keyframe batches (1–12, 7–18, 13–24…) *is* a marginalized sliding window — OKVIS2's native architecture. Confirm before building anything new.
- [ ] Marginalize old states (OKVIS2 default) vs re-linearize a covisibility window (ORB-SLAM3 style). Measure.
- [ ] Vary window length and overlap.
- [ ] Trigger a full re-linearization pass on loop closure.
- [ ] None vs VI-BA vs **rig-constrained COLMAP BA** (already working for us at 0.99 px).
- [ ] Keyframe subsampling: every 4th / 8th / 16th frame.
- [ ] Frozen vs refined intrinsics and extrinsics in the global pass.
- [ ] VIO poses as COLMAP **pose priors** vs letting it solve free.
- [ ] Iterate: VIO → BA → re-seed VIO from BA → repeat.
- [ ] Confirm non-causal output is allowed by the challenge rules (rank 2 is explicitly non-causal).

## 10. Robustness / Dynamic Content

- [ ] Masking bake-off: none / static self-occlusion mask / YOLO+SAM3 / **zero-parallax auto-detection**.
  - Zero-parallax is ours: the operator is rigidly attached, so their features have *exactly* zero parallax. Detect that geometrically, no semantic model, generalises to selfie sticks and drone frames.
- [ ] Minimum parallax / triangulation-angle gate sweep. This alone should reject rig-attached features as landmarks without any mask.
- [ ] IMU-PARSAC outlier rejection (used by the Basalt team specifically for the moving operator and flickering lights).
- [ ] Robust kernels vs masking for transient dynamics — verify the standard machinery is enough.
- [ ] Person-mask coverage stats per sequence (Hilti run_2: cam0 2.8% mean / 83% max, cam1 5.9% / 28%).
- [ ] Does masking help calibration but hurt tracking? Test the two stages independently.
- [ ] Low-light enhancement interaction with init (#14 saw CLAHE prevent init entirely).

## 11. The Bet — Rotation-First Estimation

- [ ] Estimate rotation from the **full sphere** first (globally consistent flow, no focus of expansion), then solve translation with rotation known.
- [ ] Rationale: narrow-FOV cameras confuse small rotations with small translations, and that confusion is the root of most VO drift. Over a full sphere they separate cleanly. Our gyro agrees with GT to 0.128°, so we have a strong independent check.
- [ ] Nobody in the challenge does this. Highest upside, highest risk.
- [ ] Compare against joint estimation on identical sequences.

---

## Priority Order

1. **Section 0** — deterministic mode. Nothing is measurable before this.
2. **Sections 2, 7** — equal-area bucketing, line features, angular residuals. These attack the textureless failure that beats every team.
3. **Sections 6, 4** — rolling shutter and rig handling. RS is a genuine gap in the field.
4. **Sections 8, 9** — loops and global BA. They need a good frontend to have anything to work with.
5. **Section 11** — the bet, once there's a stable baseline to compare against.
