# SLAM Candidates for Dual-Fisheye (Insta360) + IMU Rig

**Problem statement:** two ~200° fisheyes back-to-back (One RS / X-series), one IMU, no usable stereo overlap.
We need: **multi-camera monocular-rig VIO** (N=2 independent mono streams sharing one state), online **rig extrinsic refinement**, and a camera model valid **beyond 180°**.

## Camera model decision

- Pinhole + radtan dies at 180°. Kannala-Brandt (KB4/equidistant) survives >180° mathematically but **fits Insta360 lenses poorly** (user-verified).
- **Chosen model: Mei / unified camera model + radtan distortion** — Kalibr's `omni-radtan`. Better fit for these lenses; Double Sphere (DS) is the modern near-equivalent (cheaper unprojection, no iterative undistort) and is worth fitting alongside since the wide-FOV SLAM community (Basalt, OmniDSO) standardized on it.
- **Calibration route options** (pick one, can cross-check):
  1. AprilTag board video → **Kalibr** `omni-radtan` per lens + cam-IMU extrinsics + time offset in one pipeline. Needs IMU noise densities (start with generic ICM-4xxxx values, refine with Allan variance later).
  2. User-provided estimation code / model-alignment code (TBD — to be dropped in).
  3. `basalt_calibrate` for DS params (if we go the DS route for Basalt-adjacent tooling).

## Candidate table

| SLAM | Type | Multi-cam rig (no stereo)? | IMU | >180° model | Online rig refinement | What we'd have to code | Verdict |
|---|---|---|---|---|---|---|---|
| **OpenVINS** | MSCKF filter VIO | ✅ arbitrary N cams, treated independently | ✅ | KB4 only (radtan, equi) — **no Mei** | ✅ built-in: per-cam extrinsics, intrinsics, time offset (config flag) | `CamMei` class in `ov_core` (distort/undistort + Jacobians — small, clean interface), insv ingestion, per-lens masks | **PRIMARY BASE** |
| **ORB-SLAM3** | Keyframe opt. VI-SLAM | ❌ single cam (+stereo) | ✅ | KB4 (fits ok if we crop to ~160°) | ❌ | Nothing for single-lens baseline. Full rig = MultiCol-style rewrite (heavy, skip) | **Day-1 zero-code baseline** (front lens crop) |
| **Basalt** | Sliding-window opt. VIO | ⚠️ frontend is stereo-patch oriented | ✅ | ✅ DS, EUCM, UCM native (Usenko) | ❌ in VIO (calib is offline) | Restructure frontend for 2 independent mono streams — medium-heavy | Skip as base; **use `basalt_calibrate` as calibration tool** |
| **VINS-Fusion** | Sliding-window opt. VIO | ❌ mono/stereo hardcoded deep in estimator | ✅ | ✅ **Mei native** (camodocal) | ❌ | Multi-cam surgery through whole backend | Skip. (But **steal camodocal's Mei code** for the OpenVINS port) |
| **OKVIS2** | Keyframe opt. VI-SLAM + loop closure | ✅ N-cam | ✅ | radtan/equi pinhole projection — no Mei | ⚠️ extrinsics estimation supported | Add Mei projection model; insv ingestion | **Backup base** if MSCKF accuracy disappoints |
| **LF-VIO** (Zhejiang) | Opt. VIO for FOV>180° | ❌ single panoramic cam | ✅ | ✅ designed for >180° (negative-half-plane features) | ❌ | Closest prior work to our exact problem — read for the >180° feature handling tricks, esp. how they parameterize features behind the image plane | **Read + borrow ideas**, not a rig base |
| **DROID-SLAM** | Deep dense-BA | ❌ | ❌ (non-metric) | ❌ pinhole only | ❌ | Don't bolt IMU on ourselves — **DBA-Fusion (2024)** already couples IMU preintegration into DROID's dense BA as a factor graph. Feed rectified crops / cubemap faces | **Deep-learning arm via DBA-Fusion**, low priority |
| **DPVO / DPV-SLAM** | Deep sparse-patch VO | ❌ | ❌ | ❌ pinhole | ❌ | Same story as DROID, ~10× cheaper GPU. Fits 16GB VRAM easily | Alternative deep arm if DROID too heavy |
| **cuVSLAM (NVIDIA)** | Multi-cam VIO, closed source | ✅ | ✅ | fisheye4/polynomial (no Mei, can't add) | ❌ not exposed | Nothing possible — binary blob | Black-box **baseline only** |
| **Stella-VSLAM** | Keyframe VO/SLAM | ❌ (but **equirect input native**) | ❌ | equirect bypasses the issue | ❌ | Whole VI backend = too much | Vision-only arm on **our own stitcher output** (product synergy) |
| **SVO Pro** | Semi-direct sliding-window VIO | ✅ multi-cam | ✅ | ⚠️ has omni/UCM-family support | ❌ | Config + insv ingestion; frontend is brittle on rolling shutter / textureless | Dark horse; try only if OpenVINS frontend struggles |
| **MultiCol-SLAM** | Multi-fisheye keyframe SLAM | ✅ (built for exactly this geometry) | ❌ no IMU | ✅ generic omni (Scaramuzza) | ❌ | Adding IMU = building VI backend; ORB-SLAM2-era, unmaintained | Read for multi-fisheye rig formulation; don't build on it |

## Refinement stage (offline, after VIO)

1. **COLMAP 3.12 `pose_prior_mapper`** — feed VIO poses + covariances as priors on every ~8th frame. Handles the dark/dynamic-area case without touching COLMAP's backend.
2. **Custom Ceres VI-BA** — read COLMAP model into our own Ceres problem; add IMU preintegration factors between keyframes + rig constraint between the two lenses; jointly refine poses + **rig extrinsics + Mei intrinsics**. This doubles as the **per-unit calibration refiner** (premium product tie-in).
3. For non-metric deep arms: **visual-inertial alignment** (scale + gravity + biases via preintegrated IMU factors, ORB-SLAM3-style inertial-only MAP init) — NOT Sim3 against a raw IMU dead-reckoning trajectory (consumer IMU double-integration drifts meters in seconds; there is no usable IMU-only trajectory to align to).

## Test data

- **FIORD** (`Data/FIORD/`): photo-mode only — **no video, no IMU**. Vision-only + lidar-GT benchmark arm. Cannot test VIO on it.
- **Own recordings (One RS)**: 360 mode writes **two insv files** (`_00_`, `_10_`) — keep both. IMU embedded, extract with gyroflow `telemetry-parser` (or MediaSDK). Record: good light, fixed exposure, smooth motion (rolling shutter!), start=end for loop-closure metric, include one dark-hallway segment.
- **TUM-VI** (fisheye stereo + IMU, DS-calibrated) and **Hilti-Oxford** (5-cam rig + IMU): validate pipeline plumbing before trusting our own data.

## Known risks

- **Rolling shutter**: One RS video mode is RS; neither OpenVINS nor Basalt models it. Measure readout time early; keep test motion smooth; revisit if fast-rotation accuracy matters.
- **IMU noise params**: unknown for One RS. Start generic, refine with Allan variance (needs a long static recording).
- **Auto-exposure jumps** between lenses hurt cross-arm comparisons; lock exposure when possible.

## Roadmap

1. **Data**: house walk with One RS → extract IMU, verify rates/sync/timestamps.
2. **Calibration**: AprilTag board → Kalibr `omni-radtan` per lens + cam-IMU extrinsics/time offset (or user's estimation code).
3. **Baseline**: ORB-SLAM3 mono-inertial, front lens cropped ~160°, KB4. Zero code, metric reference.
4. **Main build**: OpenVINS + `CamMei` (port from camodocal) + dual-cam config + online rig refinement.
5. **Refinement**: COLMAP pose-prior mapper + custom Ceres VI-BA.
6. **Deep arm** (parallel, low prio): DBA-Fusion / DPVO on rectified crops.

All repos go under `SLAM/` as submodules-or-clones, each with its own Dockerfile.
