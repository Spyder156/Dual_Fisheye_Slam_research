# Dual-Fisheye Visual-Inertial SLAM — Problem & Method

*(status snapshot, 2026-09-05 — written for external review)*

## The problem

**Hardware.** A helmet rig with two back-to-back fisheye cameras (Insta360-class,
~190° each, KB4/equidistant calibration, 1472×1440 @ 30 Hz, monochrome) and a
990 Hz IMU. The two lenses share **no usable field-of-view overlap**, so classic
stereo is impossible: the rig is a *generalized camera* — two mono cameras with
a known 4 cm baseline and ~179.5° relative yaw.

**Task.** 6-DoF trajectory estimation on the Hilti×Trimble SLAM Challenge 2026
indoor sequences (construction sites). Score per sequence:
`S = 100·exp(−0.46·ATE)`; leaderboard #1 averages ≈96.4. Sequences contain
walk-throughs of **dark rooms** — 1–2 s stretches where one camera sees almost
nothing and the frames are motion-blur-dominated (long exposure).

**What makes it hard.**
1. Non-overlapping rig: no stereo triangulation, metric scale only from IMU.
2. Dark rooms kill corner features (0–2 ORB inliers/frame) while line features
   survive (600–1000 ELSED segments/frame) — texture is smeared, not absent.
3. Fisheye geometry: 95°+ off-axis bearings, straight world lines project to
   curves, pinhole assumptions fail everywhere.

## Our method

**Base:** ORB-SLAM3 monocular-inertial, heavily modified.

**Rig layer.** Both lenses run mono-style ORB tracking in one estimator; the
rear camera's observations are projected through the calibrated rig transform
(`T_c0_c1`, per-unit: 4.01 cm baseline, 179.56° rotation, recovered by
COLMAP-rig-constrained BA + robust median, metric scale from visual-inertial
alignment — not the assumed 180°). Landmarks are tagged by observing lens; a
single global inlier count still gates tracking (known weakness).

**Spherical line features** (the top-2 teams' signature ingredient):
- ELSED segments per lens; endpoints unprojected to unit bearings `b₁, b₂`
  on the sphere (KB4 Newton inverse).
- A segment is represented by its **interpretation plane**: the unit normal
  `n = b₁×b₂` of the great circle it paints. The measurement residual for a
  3D line `(d, m)` (Plücker: direction + moment, `m = p×d`) in a camera
  `(R,t)` is `n_c·b`, with `n_c ∝ R·m + t×(R·d)` — invariant to sliding along
  the line (the aperture problem stated honestly).
- **Matching** frame-to-frame is geometric: gates on |n·n'| (3°), chord
  direction (8°), angular-length ratio (0.5); greedy one-to-one, never across
  lenses. (An LBD appearance descriptor exists behind a flag; appearance-first
  ranking made *wrong* matches persistent and was demoted to an optional veto.)
- **Landmark lifecycle** (built to mirror what points get):
  anchor at first sight → triangulate when plane parallax ≥ 2° by
  intersecting the two interpretation planes → **extent-overlap gate** (both
  observations' endpoint rays back-projected onto the candidate line must cut
  overlapping along-line intervals — the correspondence check the
  aperture-blind residual cannot do) → **cheirality gate** (the line must
  intersect both observed bearings at positive depth; the mirror solution of a
  marginal-parallax plane intersection was 41% of creations before this) →
  2-observation probation before the optimizer may use it → extent grown by
  union over sightings → re-triangulated whenever a wider baseline appears.
- Line edges (calibrated σ≈3 px) participate in **both** inertial pose
  optimizers with the same chi2/outlier discipline as points; line landmarks
  ride every world re-expression (VIBA gravity/scale) together with points —
  ORB-SLAM3's `ApplyScaledRotation` originally moved only points, which left
  lines in a stale world frame (our single worst bug).

**Verification culture.** Every geometric stage has a falsifiable test:
pixel↔bearing round-trip (0.000 px), synthetic-truth triangulation (exact),
and an endpoint-free held-out test — every *interior* ray of every observation
must intersect the reconstructed line (measured 1.2 cm median on a 61-frame
track). Endpoint identity is never assumed: detected endpoints slide along the
physical edge between frames (clipping), so endpoints are treated as
"samples on the line", never as repeatable keypoints.

## Where we are

- Best runs: **84–85 score** on the hardest sequence (run_2), surviving dark
  room 1 with ~1 m damage; earlier three-sequence mean 78.4 with a weaker rig.
- Line landmarks are geometrically sound (cm-level on long tracks) and are in
  the same frame as points (verified visually and numerically).

## The two open problems

1. **Run-to-run variance.** Identical binary/config/data scores 85 or 0.4
   depending on thread interleaving: ~10% of runs cross dark room 1. Frontend
   is deterministic (identical per-frame detection/match counts across runs);
   the divergence is backend state (keyframe/BA timing) entering the dark
   window. Survivors log ~6 tracking failures; corpses 200+. (Notably, the #3
   team — OKVIS2-X's own authors — reported *best of six runs*, acknowledging
   the same nondeterminism.)
2. **Line track length.** Median line track ≈ 2.3 observations. ELSED
   re-fragments segments nondeterministically frame-to-frame; a one-to-one
   segment matcher with a length-ratio gate loses a track whenever a segment
   splits, so most landmarks triangulate at the 2° parallax floor and carry
   ~10–20% depth noise (the "confetti map"). Long tracks are cm-accurate — the
   estimator is starved, not wrong. Planned fix: match *plane clusters*
   (collinear segments merged per frame) instead of raw segments, making the
   matching unit invariant to fragmentation and endpoint clipping.

## Response to external review (six suspects, 2026-09-05)

1. **Chord-direction gate — CONFIRMED WRONG by algebra.** With bearings
   parametrized by arc angle from the foot point, the chord is the great-circle
   tangent at the segment *midpoint*: chord∠d = |α_mid|. Our 15° gate therefore
   tested where the segment sat on the circle, not whether `d` was right
   (1.9M rejections logged). **Replaced** everywhere with the extent-overlap
   test: both observations' endpoint rays back-projected onto the candidate
   line must cut overlapping along-line intervals.
2. **No map→frame re-acquisition — confirmed, built.** Landmarks are now
   projected into each frame (predicted great circle + extent) and any unbound
   segment within 3° of plane with overlapping extent binds, many-to-one
   (legitimate: the residual is aperture-blind, each fragment is an independent
   plane constraint). First harness run: 346k re-acquisitions / 30 s, 0 resets.
   Fragmentation and clipping stop mattering; blind-frame re-attachment exists.
3. **Optimizer convention — structurally cleared.** `ImuCamPose::Update`
   refreshes `Rcw[i]` for *all* cameras per iteration; the line edge has no
   `linearizeOplus`, so g2o numeric-differentiates through that exact oplus
   (right-perturbation convention consistent by construction). A lines-only
   synthetic convergence test remains queued.
4. **Keyframe thread lottery — REFUTED by the decisive experiment.** LOCKSTEP
   mode (tracking drains LocalMapping every frame) run 3-vs-3 against free
   threading, fixes 1–2 included: lockstep 0.15/0.00/0.12 vs free
   0.40/0.07/0.04 — determinism does not rescue the crossing. The
   landmark-backed bet also resolved: failing runs carry **230–314**
   landmark-backed line edges per dark frame and die regardless. Lines are
   present and voting through the dark window — which moves suspicion to #5:
   blur-biased line measurements at full weight may be actively harmful there.
5. **Blur-scaled σ / exposure-offset** — queued (needs residual-vs-ω
   instrumentation on dark frames).
6. **T_c0_c1 orientation sign** — partially screened (cam0/cam1 line chi2
   medians equal, which a ~7 px systematic would skew); signed-residual
   histogram queued.

## Reference points (challenge leaderboard)

- **#1** ACDC-VSLAM: same spherical-line formulation over ELSED; angular-length
  filtering; BA reweighting by track length/inverse depth.
- **#2**: PL-VIWO-style **structureless** line updates (MSCKF nullspace, no
  line landmarks at all) + weak-Manhattan vertical prior; cross-camera
  short-window matching for local metric scale.
- **#3** (OKVIS2-X authors): EUCM, online extrinsics, medianBlur for dark
  scenes, best-of-6 reporting.
