# SLAM Experiment Log — Insta360 One RS dual-fisheye VIO

Running log of every experiment, result, and open question. Newest at the bottom.
Visualizations live in `outputs/slam_debug/` and `Data/Home/dataset/` (user-verified ones noted).

## Dataset: Data/Home/dataset (house walk, 2026-08-13)

- Source: `VID_20250611_064750_{00,10}_062.insv`, 85.2s, 2944×2880 @ 29.97fps per lens.
- `cam0/`, `cam1/`: 2553/2552 JPG frames, rotated 180° (camera held inverted — **user confirmed** upright after fix).
- `imu.csv`: 86,000 samples @ 997.6Hz, gyro rad/s, accel m/s² (|a|≈9.72 ✓).
- `equirect_sdk.mp4`: MediaSDK optflow+flowstate 5760×2880 (stabilized — do not use for SLAM).
- No ground truth; loop-closure error is the metric (walk starts/ends at same spot).

## Factory calibration (extracted from insv trailer)

- Mei model per lens: xi=2.0149, fx≈2989.6/2979.8 + 5 distortion coeffs; poly model as cross-check.
- Rig: back lens ≈ 180° roll + 4.0cm z-offset. Distortion order `k1k2p1p2` (user picked top-2 rows
  of `calib_validation.png`; SDK photometric cross-check inconclusive, NCC only 0.40 — likely frame
  timing; not pursued further per user).
- Mapped to rotated video frames: centers point-reflected, tangentials negated. See `scripts/factory_calib.py`.

## Experiment log

| # | What | Result | Verdict |
|---|------|--------|---------|
| 1 | OpenVINS + CamMei port, dual-cam, factory calib, T_imu_cam0=identity guess | Ran 2552 frames, filter alive, but 5.4 km "trajectory" in 85s; ba blew up to ~0.8 | Diverged — extrinsics guess wrong |
| 2 | Gravity check from accel | Gravity on IMU **+x** the whole walk → identity guess impossible; 4 right-handed candidates remain | Root cause #1 found |
| 3 | 4 extrinsic candidates (A–D), 35s slice | ALL diverge (313–701m path). D least bad | Extrinsics not the only problem |
| 4 | CamMei numeric validation (roundtrip + FD Jacobians) | Machine precision at θ=5–80° | Model code is correct |
| 5 | IMU↔frame time alignment (gyro vs frame-diff cross-correlation) | Offset ≈ −0.01s, corr 0.56 | Timestamps fine — hypothesis killed |
| 6 | DEBUG-log analysis of run D | Static init fired **mid-walk** with v=0 (poisons ba); **86% of MSCKF updates get 0 features**, rest 1–7. SLAM updates get 1–5 (so triangulation path works) | Root cause #2: init; root cause #3: feature starvation |
| 7 | Fix init (thresh 1.5, disparity 4.0, clones 15), re-run D | Still diverges (1078m) — starvation dominates | Init fix necessary but insufficient |

## Current blocker: feature starvation

MSCKF receives 0 features in 86% of updates → filter dead-reckons on IMU → velocity runaway.
Tracking montage (`Data/Home/dataset/tracking_montage.png`) reviewed by user: **not informative** — need
better diagnostics, not more montages.

Candidate explanations (NOT yet tested — awaiting discussion):
1. KLT tracks live too long/too stably → features rarely "die" → never handed to MSCKF; with
   max_clones exhausted they should still be consumed at window edge — needs instrumentation of
   `feats_lost` / `feats_marg` counts per frame.
2. Triangulation rejections inside UpdaterMSCKF (silent) — chi2 warnings were 0, so death is
   before/at triangulation, not at the gate.
3. Wide-FOV normalized coords (|zn| up to ~4.7 near mask edge) breaking FeatureInitializer
   thresholds (`max_cond_number`, `lam_thresh` defaults).
4. Update happens only when features die at the same time the anchor clone marginalizes —
   config interplay of max_clones vs track length.

## Visualizations index

- `outputs/slam_debug/01_trajectories_divergence.png` — A–D top-down + speed runaway profile
- `outputs/slam_debug/02_feature_starvation.png` — feats/update timeline + histogram
- `outputs/slam_debug/03_imu_overview.png` — accel (gravity on x) + gyro profile
- `Data/Home/dataset/preview_montage.png` — dataset sanity (user-approved)
- `Data/Home/dataset/calib_validation.png` — distortion-order rows (user: top 2 plausible)
- `Data/Home/dataset/tracking_montage.png` — KLT tracks (user: not telling)
