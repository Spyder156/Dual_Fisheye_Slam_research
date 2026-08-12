# Ideas: Technical & Research Directions

**Companion to:** sfm-grade-stitching-project.md
**Organization:** Tier 1 (proven, ship-critical) → Tier 2 (established but non-trivial) → Tier 3 (novel/research-grade, high risk, high differentiation)

Impact-ordered priority across all tiers (from discussion):
**entrance pupil drift → per-sensor RS direction/delay → overlap ray-consistency self-calibration → residual spline in (θ,φ) → chromatic aberration per-channel → thermal terms.**
The first three attack systematic errors that alias into pose; the rest are refinement.

---

## TIER 1 — Foundation: Proven, Must-Have
*Standard techniques with literature support. Low risk. These make the product work at all.*

### 1.1 Base parametric lens model: Mei omni + radtan
- Converges well at ~200° FOV (empirically confirmed in own experiments) and unifies both lenses under one model family.
- Fallback: Double Sphere (robust ultra-wide, cheap unprojection — see Usenko et al., 3DV 2018 for unprojection cost analysis). Kannala-Brandt deprioritized (failed convergence at 200° in own Kalibr tests).
- **Paper support:** Garcia et al. (Photogrammetric Record 2024) showed a rigorous lens-specific projection model (equisolid) outperformed generic EUCM on a real dual-fisheye (Ricoh Theta S) 140 m trajectory → lens-appropriate/refined models measurably beat generic ones.

### 1.2 Higher-order radial terms + solid-angle-weighted calibration residuals
- Standard radtan k1–k3 fit is dominated by the image center where checkerboard corners land; the >90° periphery (seam-critical zone) is statistically underfit even when global RMSE looks good.
- Add k4–k6 or rational p(θ)/q(θ), and **weight calibration residuals by solid angle, not pixel count**.

### 1.3 Rigid lens-to-lens extrinsics as a hard BA constraint
- 6-DoF lens-to-lens transform, factory values as initialization only (they drift with temperature/impacts).
- **Paper support:** Perfetti/Polari/Fassi (ISPRS 2018): rigidly-constrained multi-camera calibration improved accuracy *to millimetres* vs. auto-stitched equirectangular. Barazzetti et al. (2022): manufacturer equirect has low metric accuracy; per-lens calibration + custom re-stitch recovers it. → The core product thesis, already validated academically, never productized.

### 1.4 Camera–IMU temporal calibration
- IMU-camera time offset + rolling-shutter line delay via Kalibr camera-IMU stage.
- Dominates the video error budget more than lens-model choice. First-class citizen from day one.

### 1.5 Photometric calibration
- Per-lens vignetting map, sensor response (EMoR or gamma+gain), and **full 3×3 inter-sensor color matrix** (not per-channel gain — the two ISPs don't share white-balance behavior).
- Required for photometric losses, direct SLAM, and clean GS training.
- **Paper support:** Omnidirectional Dense SLAM for Back-to-back Fisheye Cameras explicitly builds photometric compensation between the two views.

### 1.6 Validity masks instead of parallax fill
- ~2 cm baseline → no single center of projection → mask invalid pixels honestly; per-pixel validity channel ships with every frame.
- Diffusion infill only as a strictly separated visualization export with fill masks (hallucinated texture = hallucinated features in SfM).

### 1.7 Spherical-domain evaluation & RANSAC correctness
- Essential-matrix / inlier thresholds must be set in the spherical domain, not pixel space — tie-point distribution on the sphere changes the correct threshold (PAN-SLAM, Wuhan Univ.).
- Adopt RayTun3R's metric protocol (R°, t°, d_reproj on TUM-VI / FIORD / ScanNet++) so results are directly comparable to the DL literature.
- External validation options: AprilTag room, laser-distance control points (tunnel-mapping protocol, Remote Sensing 2022), lidar GT (360Loc methodology; FIORD dataset, CC BY 4.0).

---

## TIER 2 — Differentiators: Established Physics, Rarely Modeled
*Known effects with niche literature. Medium risk. Nobody in consumer 360 does these; they're the accuracy moat.*

### 2.1 Per-sensor rolling shutter with opposite read directions ★ (top-3 priority)
- The two sensors are mounted opposed → RS scan directions differ in world frame. A single line-delay parameter is wrong.
- Model line delay **per sensor** plus read-direction geometry in RS correction.
- Failure signature if unmodeled: velocity-dependent seam mismatch that masquerades as extrinsic error.

### 2.2 Per-sensor trigger time offset
- The two ISPs may not trigger simultaneously. Sub-millisecond, but = pixels at the periphery under fast rotation. Learn one offset per sensor, not one global camera-IMU offset.

### 2.3 Thermal calibration coefficients
- Focal length / principal point drift with temperature on plastic-barrel consumer lenses; Insta360 logs temperature in metadata.
- Make intrinsics affine in T: f(T) = f₀ + α_f·ΔT. Extend to extrinsics (chassis flex): 6-DoF base + small thermal linear term.
- Matters for surveyors running 40-minute captures. Nobody in the consumer 360 world does this.

### 2.4 Chromatic aberration as three sub-models
- Lateral CA at 95–100° incidence is multiple pixels on these lenses. Calibrate per-channel distortion deltas (R, B as offsets from G); extract features on G or CA-corrected image.
- Significant for photometric losses and GS training; almost universally ignored.

### 2.5 Azimuthal (non-radial) distortion: low-order Fourier in φ per radial band
- Cheap molded aspheres have azimuthal distortion asymmetry (distortion depends on φ, not just θ); radtan tangential terms capture only the first harmonic.
- A handful of Fourier coefficients per radial band; interpretable, low-dim, and a per-unit manufacturing signature → strengthens the self-calibration moat.
- (Overlaps with what a residual spline can absorb — pick one; Fourier is more interpretable.)

### 2.6 Peripheral calibration methodology for 190–200°
- Checkerboards can't reach θ > ~80° reliably (grazing angles, blur). Fixes:
  - Target-lined room / AprilTag dome so targets appear at true peripheral angles.
  - **Line-based (vanishing-point) constraints**: straight structures imaged at extreme θ give pure distortion constraints without corner detection — an old photogrammetry trick, mostly forgotten.
- Network design guidance: Sensors 2025 fisheye calibration network-design study → feeds the customer-facing guided-calibration procedure.

### 2.7 Depth-conditioned seam placement (the only permitted content-dependence)
- Which lens supplies which pixel in the overlap may be depth-conditioned; the warp itself never is.
- Placement refinement: put the seam where downstream gradients are healthiest — PFGS360 (2603.23324) shows omnidirectional 3DGS Jacobians amplify errors near ERP poles; weight pose losses by solid angle for the same reason.

### 2.8 Calibration initializers from single images
- AnyCalib (ICCV 2025) / PRaDA (CVPR 2025) as coarse κ initializers for self-calibration; your BA refines. They are initializers, not competitors — RayTun3R's own Table 5 shows predicted calibration measurably degrades results vs. GT calibration → refinement retains value.

---

## TIER 3 — Research Frontier: Novel, High Risk, High Differentiation
*Little or no direct literature for this exact regime. Paper-shaped. This is where the durable moat and publishable novelty live.*

### 3.1 Entrance pupil drift modeling ★ (highest-impact single idea)
- Ultra-wide fisheyes have a non-stationary entrance pupil: the effective projection center moves along the optical axis as a function of incidence angle (up to several mm at 95–100°).
- Model as learnable low-order polynomial d(θ) shifting the ray origin (cf. Gennery's generalized fisheye model; wide-FOV metrology literature).
- **Why it's critical here:** at the seam (θ ≈ 95–100°), unmodeled pupil drift systematically corrupts lens-to-lens extrinsics — the optimizer smears pupil drift into a wrong baseline. Likely worth more than any added distortion term.

### 3.2 Residual warp field in (θ, φ) on top of the parametric prior
- B-spline grid parameterized in (θ, φ) — not pixel coords — with knot density increasing toward θ = 90–100° (uniform pixel grids waste capacity in the center, starve the periphery).
- Schöps-style generic-model hybrid: parametric model authoritative below ~80°, spline takes over the last 10–20°. Regularize with smoothness + zero-mean constraint so the spline can't absorb extrinsics.
- **Upgrade path:** tiny SIREN/MLP over (θ, φ) as the residual — smooth by construction, ~1–2k params, composes cleanly with backprop ("neural lens model" as residual, keeping parametric interpretability).
- **Alternative representation worth stealing:** spherical-harmonics-over-rays (UniK3D → Wid3R → CAM3R lineage) — SH ray fields are becoming the DL-world's standard camera representation; consider exporting calibration *as* an SH ray field (third export format alongside COLMAP and ERP).

### 3.3 Cross-lens overlap as free self-calibration supervision ★ (top-3 priority; backbone of the product moat)
- In the ~10–20° overlap band both lenses see the same rays → epipolar/ray-consistency constraints couple both intrinsics + extrinsics *exactly at the angles that matter most*, on every frame of arbitrary user footage.
- Combine with IMU rotation priors: pure-rotation segments make the overlap constraint nearly depth-free → peripheral distortion refinable from casual footage, no targets.

### 3.4 Pose-distillation fine-tuning loop (Stage C)
- Pseudo-GT poses from raw dual-fisheye frames via existing high-accuracy pose model (t_err < 0.1°, R_err < 0.1°) → stitch with current parameters → pose estimation on the ERP output → loss vs. pseudo-GT + reprojection residuals + photometric consistency → backprop through the differentiable stitch to refine parameter groups.
- Plain gradient descent / Ceres joint optimization — **no RL** (every term differentiable or has analytic Jacobians; cf. Liu et al., Sensors 2019 for analytic reprojection Jacobians on 195° fisheye with EUCM — derive the Mei+radtan equivalents).
- Circularity guard: fixed external pose targets (structurally the same trick as RayTun3R's MAGSAC++ pose pseudo-labels computed once by an external matcher, so the adapter can't influence its own labels).
- Second pose source for disagreement-flagging: fisheye-adapted foundation models (RayTun3R/Fisheye3R-adapted VGGT/π³/DA3) as an independent learned pose estimate.

### 3.5 Per-pixel expected-error output
- Beyond binary validity masks: per-pixel expected reprojection error given a depth prior. Downstream SfM/GS can weight by it. Stretch goal; no direct precedent found in the 360 product space.

### 3.6 Calibration inside the renderer (Seam360GS-style) as a cross-check
- Seam360GS (ICCV 2025) jointly optimizes lens distortion + inter-camera gap (±2 cm simulated) via GS rendering loss — structurally identical to Stage C but with photometric-rendering supervision instead of pose supervision.
- Idea: use a GS-rendering-loss refinement as an *independent validation* of the calibration recovered by the pose-loss pipeline. Two supervision signals agreeing = strong per-unit confidence metric (also a customer-facing QA number).

### 3.7 The paper-shaped novelty (identified combination)
- **"Joint entrance-pupil + azimuthal-Fourier + overlap-constrained self-calibration for dual back-to-back fisheye rigs"** — no known publication combines these, and θ≈95–100° on a two-lens consumer rig is precisely the regime where each term matters most.
- Second paper-shaped gap (from literature survey): a rigorous quantified analysis of how content-adaptive stitching corrupts SfM/SLAM — the Phase 0 benchmark, written up, fills a hole three independent groups (Barazzetti 2022, CEJ 2023, FullCircle 2026) have gestured at but never isolated. A workshop paper here doubles as the marketing asset.

### 3.8 Training-data generation as a second product surface
- The 360-depth literature (PanoFormer/EGformer/SGFormer) explicitly complains that GT equirectangular depth is extremely scarce.
- Calibrated pipeline + a depth sensor = a generator of metrically-correct ERP+depth training data — potential future revenue line serving the very research community currently trending toward bypassing the stitch.

---

## Cross-cutting design rules (from the whole discussion)
1. **The stitch is a fixed function.** All learning = refining parameters of one ray-consistent mapping. Content-adaptive warping is the disease, not the cure. Only seam *placement* may be content/depth-dependent.
2. **The calibration is the product; equirect is one export.** Ship calibrated dual-fisheye + camera models (COLMAP/OpenSfM/ORB-SLAM3 configs, optionally SH ray fields) as first-class outputs — this hedges the "foundation models eat the stitch" trend, since every adaptation method (RayTun3R included) still consumes a camera model κ as input.
3. **Benchmark before building** (Phase 0), using protocols directly comparable to both the classical (COLMAP metrics, laser control points) and DL (R°/t°/d_reproj on FIORD) literatures.
