# SfM-Grade Stitching Pipeline for Insta360 Cameras

**Project codename:** TBD
**Author:** [You]
**Status:** Specification v0.1
**Date:** July 2026

---

## 1. Project Goal

Build a stitching pipeline for Insta360 dual-fisheye cameras that produces **geometrically correct, ray-consistent equirectangular output optimized for SfM / SLAM / photogrammetry** — not visual seamlessness.

MediaSDK's stitcher optimizes for how the output *looks*: content-adaptive, flow-based warping that hides the seam but destroys metric geometry, and does so differently on every frame. This is the root cause of drift, broken feature tracks, and inconsistent reconstructions reported by robotics and 3D scanning teams using Insta360 hardware.

This project inverts the objective:

- **One fixed, calibrated, ray-consistent mapping** from dual fisheye to equirectangular. Same geometry every frame.
- **Honest invalid-pixel masks** instead of hallucinated or warped fill. Parallax dead zones near the seam are masked, not hidden.
- **Per-unit calibration refinement** so the mapping reflects the physical camera in hand, not factory nominals.
- **Documented camera model** that downstream tools (COLMAP, OpenSfM, ORB-SLAM3, Gaussian splatting pipelines) can consume with confidence.

### Non-goals

- Competing with MediaSDK on visual seamlessness for consumer content.
- Content-adaptive warping of any kind in the geometry path.
- Real-time performance in v1 (offline processing first; real-time is a later milestone).

---

## 2. Core Technical Principle

**The stitch is a fixed function, not a learned image-to-image model.**

For each equirectangular pixel: ray → lens selection → projection through calibrated lens model → sample. All "learning" in this project means **refining the parameters of that fixed mapping**, never making the warp content-dependent.

The one permitted content-dependence: **seam/mask placement** (which lens supplies which pixel in the overlap region) may be depth-conditioned. The warp itself never is.

### Why the black regions are a feature

The two lenses have a ~2 cm baseline — there is no single center of projection. A truly geometrically correct equirect is physically impossible for nearby content. Rather than faking it:

- Pixels with no valid single-ray interpretation are **masked out** (black + binary mask channel).
- Per-pixel validity masks ship with every frame.
- Stretch goal: per-pixel expected reprojection error given a depth prior.

Sophisticated SfM customers value honest masks over pretty pixels. Feature extractors respect masks; dense stereo respects masks; hallucinated texture generates hallucinated features.

### Optional visualization export (strictly separated)

A diffusion-based infill may be offered as a **separate "visualization export"** — never in the geometry path, always accompanied by the fill mask so downstream photogrammetry can exclude filled regions. Two outputs, two jobs.

---

## 3. Camera & Calibration Model

### 3.1 Lens model

- **Primary candidate:** Mei omnidirectional + radtan distortion. Empirically converges well at ~200° FOV and unifies both lenses under one model family.
- **Fallbacks:** Double Sphere (cheap unprojection, robust ultra-wide), Kannala-Brandt (noted: failed to converge at 200° FOV in prior Kalibr experiments — deprioritized).
- **Headroom over parametric models:** low-dimensional residual warp field per lens (B-spline grid or small MLP over fisheye image coordinates), capturing lens behavior the parametric model can't. Parametric model acts as the prior; residual field is learned per-unit. (Cf. generic camera models, Schöps et al.)

### 3.2 Full learnable parameter set

| # | Group | Parameters | Notes |
|---|-------|-----------|-------|
| 1 | Intrinsics corrections | δ focal, δ principal point, δ distortion coeffs per lens; optional residual warp field | Factory values as initialization |
| 2 | Extrinsics | 6-DoF lens-to-lens transform | Factory values approximate; drift with temperature/impacts |
| 3 | Temporal | IMU–camera time offset, rolling-shutter line delay, per-frame gyro bias | **Dominates video error budget** — first-class citizen from day one |
| 4 | Photometric | Vignetting map per lens, exposure/gain offset between sensors | Required for photometric losses and direct SLAM |
| 5 | Seam policy | Mask boundary placement, optionally depth-conditioned in overlap | Only permitted content-dependence |

Roughly 50–200 parameters in the parametric case; more with residual warp fields.

### 3.3 Calibration pipeline

**Stage A — Lab calibration (own units):**
Kalibr with checkerboard/AprilGrid targets. Camera-only intrinsics + extrinsics, then camera–IMU joint calibration (time offset, line delay).

**Stage B — Per-unit refinement (the product differentiator):**
Customers' units differ from yours. Two delivery modes:
1. **Self-calibration from user footage:** joint refinement of parameter groups 1–4 via bundle-adjustment-style optimization on customer video.
2. **Guided calibration procedure:** customer films a specified target/motion pattern; pipeline returns a per-unit calibration file.

**Stage C — Pose-supervised fine-tuning (distillation):**
- Extract pseudo ground-truth poses directly from **raw dual-fisheye frames** using the existing high-accuracy pose model (t_err < 0.1°, R_err < 0.1°).
- Forward pass: stitch with current parameters → equirect.
- Run pose estimation (PnP + reprojection) on the stitched equirect.
- Loss: pose discrepancy vs. dual-fisheye pseudo-GT + reprojection residuals + photometric consistency.
- Backprop through the (differentiable) stitching function to refine parameter groups 1–4.
- Plain gradient descent / Ceres-style joint optimization. **No RL** — every term is differentiable or has analytic Jacobians.

External validation (guarding against circularity): AprilTag grid room, known-geometry rig, or laser-scan reference scene.

---

## 4. Pipeline Architecture

```
.insv/.insp file
   │
   ├─► Parser: raw dual-fisheye streams, factory calib blob,
   │           gyro/IMU stream, per-frame metadata
   │
   ├─► Sync: frame timestamps ↔ IMU (time offset from calibration)
   │
   ├─► Per-unit calibration (Stage B/C output): intrinsics,
   │           extrinsics, temporal, photometric parameters
   │
   ├─► Rolling-shutter-aware reprojection:
   │       equirect ray → lens selection → RS-corrected projection
   │       → sample (with photometric harmonization)
   │
   ├─► Outputs:
   │       • Equirect video/frames (fixed ray-consistent mapping)
   │       • Per-pixel validity mask stream
   │       • Camera model file (COLMAP/OpenSfM-compatible spec)
   │       • [Optional, separate] diffusion visualization export + fill masks
   │
   └─► (Alternate output) Calibrated dual-fisheye + camera models,
           for customers who skip equirect entirely
```

Note the alternate output: some customers will run dual-fisheye SLAM directly. **The calibration is the product; equirect is one export format.**

---

## 5. Roadmap

Each phase ends with a **verifiable result** and a **visualization artifact**. No phase is "done" without its artifact.

### Phase 0 — Benchmark First (2–3 weekends) ⚠️ DO THIS BEFORE MORE ENGINEERING

**Goal:** Quantify the gap. This chart is simultaneously demand validation, the sales deck, and the go/no-go decision.

- Capture 3–5 scenes (indoor structured, outdoor, mixed depth) with one Insta360 unit.
- Stitch each with (a) MediaSDK, (b) current in-house stitch.
- Run COLMAP on both. Also run a spherical-camera SLAM (e.g., OpenVSLAM-lineage / Stella-VSLAM equirect mode).
- **Metrics:** reprojection RMSE, mean feature track length, pose ATE/RPE (vs. AprilTag or loop-closure reference), dense reconstruction completeness.
- **Artifacts:** one comparison chart per metric; side-by-side sparse point clouds; track-length histograms.
- **Go/no-go:** if the delta is small, stop and save six months. If big, this chart is the pitch.

### Phase 1 — Parsing & Deterministic Stitch Core (4–6 weeks)

- Robust .insv/.insp parser: streams, factory calibration blob, IMU, metadata. Version-tolerant design (firmware format changes are an ongoing tax — isolate behind an interface).
- Fixed-mapping reprojection engine (CPU reference implementation, then GPU).
- Validity mask generation.
- **Verifiable result:** bit-exact reproducibility — same input, same output, every run. Reprojection self-consistency test: project known 3D points through the model, verify pixel positions.
- **Artifact:** stitched equirect + mask overlay video from raw .insv, no MediaSDK anywhere in the loop.

### Phase 2 — Lab Calibration (3–4 weeks)

- Kalibr pipeline: Mei+radtan primary, Double Sphere fallback. AprilGrid captures.
- Camera–IMU joint calibration: time offset + rolling shutter line delay.
- **Verifiable result:** reprojection RMSE < 0.5 px on held-out calibration frames; time offset repeatability < 0.5 ms across sessions.
- **Artifact:** distortion residual heatmaps per lens (parametric model error map — this also sizes the residual-warp-field headroom).

### Phase 3 — RS-Aware Stitching + First Geometry Win (4–6 weeks)

- Integrate calibration + gyro into RS-corrected reprojection.
- Photometric harmonization (vignetting, exposure offset).
- **Verifiable result:** re-run Phase 0 benchmark with the new pipeline. Target: measurable improvement over MediaSDK on ATE and track length on all scenes.
- **Artifact:** THE chart — MediaSDK vs. ours, COLMAP metrics side by side. This is the landing page.

### Phase 4 — Per-Unit Self-Calibration (6–8 weeks, the product moat)

- Joint BA-style refinement of parameter groups 1–4 from arbitrary user footage.
- Pose-distillation fine-tuning loop (Stage C) as refinement layer.
- Residual warp field experiments (does it buy measurable SfM improvement over parametric-only?).
- **Verifiable result:** take a *second* unit (never lab-calibrated), self-calibrate from footage only, match within 20% of lab-calibrated benchmark performance. Validate against AprilTag room.
- **Artifact:** convergence plots; before/after reconstruction of the same scene with factory vs. refined parameters.

### Phase 5 — Productization (4–6 weeks)

- Camera model export: COLMAP / OpenSfM / ORB-SLAM3 config formats.
- CLI tool + library API. Docs.
- Multi-model support matrix (X3/X4/X5 — verify parser + calibration per model).
- Optional: diffusion visualization export (separate module, fill masks mandatory).
- **Verifiable result:** an external user goes from .insv to a COLMAP reconstruction using only the docs.
- **Artifact:** end-to-end demo video; public benchmark page.

### Phase 6 — Ongoing

- Firmware/format regression suite (every Insta360 firmware release → parser CI).
- Temperature drift study (calibration stability across thermal range).
- Real-time/GPU streaming variant for robotics customers (if demand pulls it).

**Total to sellable v1: ~5–7 months of side-hustle time**, with a go/no-go checkpoint at week 3.

---

## 6. Evaluation Protocol (used in every phase)

**Geometric metrics (primary):**
- Reprojection RMSE (calibration scenes, known targets)
- Feature track length distribution (COLMAP)
- Pose ATE / RPE vs. reference (AprilTag room, loop closures, or laser scan)
- Sparse map consistency across repeated passes of the same trajectory
- Dense reconstruction completeness & accuracy vs. reference geometry

**Consistency metrics:**
- Frame-to-frame mapping stability (the whole point vs. MediaSDK: warp field variance over time = 0 by construction; verify it)
- Cross-unit repeatability after self-calibration

**Photometric metrics (secondary):**
- Seam-region exposure discontinuity
- Vignetting residual after correction

**Standard visualizations for every experiment:**
- Side-by-side sparse clouds (MediaSDK vs. ours)
- Track-length histograms
- Trajectory plots with GT overlay
- Distortion/error heatmaps in fisheye and equirect domains
- Mask coverage statistics per scene depth profile

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Insta360 ships calibrated/SfM output natively | Existential | Move fast; build per-unit calibration + benchmark credibility they can't quickly replicate; niche is likely too small for them to prioritize |
| Firmware format changes break parser | Ongoing tax | Version-tolerant parser design; regression suite per firmware release; format knowledge is itself a moat |
| Customers skip equirect, use dual fisheye directly | Partial | Already handled: calibrated dual-fisheye + camera models is a first-class output. Sell the calibration, not the stitch |
| Self-calibration fails on low-texture user footage | Product quality | Guided calibration procedure as fallback; footage-quality pre-check in tool |
| Legal exposure from reverse engineering | Legal | No MediaSDK binary reverse engineering. File-format parsing from first principles + open-source parsers only |
| Parallax dead zones unacceptable to some buyers | Market | They were getting silently corrupted geometry before; masks + optional visualization export address both audiences |

---

## 8. Business Model (summary — see ongoing discussion)

- **Wedge:** calibration-as-a-service (~$200–500/unit) — validates demand, near-zero incremental build cost.
- **Core:** B2B SDK/library annual license, $5k–50k/yr by company size. Buy-vs-build math strongly favors buying: 6 months of a CV engineer ($80–150k US / $30–50k offshore) + permanent maintenance vs. a license.
- **Later:** desktop tool subscription ($50–150/mo) for surveyors/photogrammetrists; possibly cloud API.
- **Target market:** robotics companies, scanning/surveying firms, NeRF/3DGS pipeline teams using Insta360 rigs. Est. 50–500 serious organizations worldwide. 10–20 SDK customers at $10–20k/yr → $100–400k/yr. Lifestyle-scale, defensible, deliberately below big-competitor radar.
- **Trust builder in pitch:** demonstrated ability to replicate MediaSDK-quality stitching (99.6% photometric match) independently — proves format/pipeline mastery. Lead with the *geometric* benchmark, use photometric parity as capability proof.

---

## 9. Immediate Next Actions

1. [ ] Capture Phase 0 benchmark scenes (3–5 scenes, varied depth profiles)
2. [ ] Set up AprilTag reference room / target rig
3. [ ] Run MediaSDK + in-house stitch → COLMAP → metrics
4. [ ] Produce THE chart
5. [ ] Go/no-go decision
6. [ ] If go: share chart with 3–5 target companies, gauge prepay interest before Phase 1 completes
