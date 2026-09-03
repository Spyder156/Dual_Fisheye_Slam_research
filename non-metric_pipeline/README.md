# non-metric_pipeline

Dual-fisheye SLAM for **Insta360-class cameras in the real world** — not the
Hilti challenge. Separate from `SLAM/` on purpose: different goal, different
trade-offs, and it must not drift as the challenge work churns.

## What "non-metric" means, and why it is the right call here

The challenge is scored on **metric** ATE, so every centimetre of scale error
costs points. A product usually does not need that:

| needs metric scale | does not |
|---|---|
| measuring rooms, volumes, distances | camera path for stabilisation / reframing |
| AR object placement at true size | 3D reconstruction for viewing |
| survey / as-built | relocalisation in a previously mapped space |

Metric scale is the single hardest thing to get right from a monocular rig — on
our own data it drifted 0.99 → 1.68 within one sequence, and neither a global
nor a windowed correction fixed it reliably. Dropping the requirement removes
the least reliable part of the system.

**What we keep, which is scale-free and does work:**
- rig ROTATION (recovered to ~0.03 deg)
- trajectory SHAPE (Sim(3) ATE 0.087 m on Hilti run_1 — rank-1 quality)
- the map, up to a similarity

If a metric anchor is available later (IMU with good excitation, a known
distance, a measured object), it applies as a single similarity at the end
without touching anything upstream.

## Stages

```
  0  per-unit calibration      COLMAP, from the video itself
  1  SLAM                      ORB-SLAM3 mono-rig, both lenses, one map
  2  refinement                bundle adjustment over the finished trajectory
```

### 0 — per-unit calibration (the part nobody else does)

Every Insta360 body is slightly different, and the factory calibration does not
describe the unit in your hand. Recovered from a short clip of the actual
device:

- per-lens intrinsics (focal, principal point, distortion)
- **inter-lens angle** — everyone assumes 180.000 deg; it is not. On our
  reference rig the truth was 179.561, and assuming 180 costs 0.44 deg
- **inter-lens baseline** — everyone assumes zero; it is ~4 cm

The trick that makes the rig observable at all with non-overlapping lenses:
walking forward, the REAR lens at t+dt sees what the FRONT lens saw at t. So
match `front[i] <-> back[i+dt]` over a set of offsets. No SLAM needed, no
calibration target, no ground truth.

Measured against a rig with known truth: angle to **0.03 deg**, baseline to
**~1%**.

### 1 — SLAM

ORB-SLAM3 on the MONOCULAR path with both lenses feeding one Frame:
- descriptors pooled across lenses (also what cross-lens loop closure needs)
- no cross-camera stereo matching — the lenses share no field of view
- rig extrinsics FIXED from stage 0, so there is no cold start

Why both lenses matter, measured on a sequence where the wearer walks into a
wall and the front lens loses all texture:

| | score |
|---|---|
| front lens only | 45.4 |
| both lenses, correct extrinsics | **84.7** |
| both lenses, zero baseline assumed | 5.3 |

Note the third row: a rig with the WRONG baseline is far worse than no rig at
all. The extrinsics are not a detail.

### 2 — refinement

Bundle adjustment over the completed trajectory. Structure is triangulated into
the SLAM poses, then poses and structure are optimised together.

Caution learned the hard way: if the structure is triangulated from the very
poses the BA then moves, the BA lowers reprojection error while making the
trajectory worse. Refinement needs information the front end did not already
have — loop closures, or an independent reconstruction.

## Status

Working: stage 0, stage 1.
Stage 2 is wired but needs independent structure to be worth running.

## Not included, deliberately

- metric scale recovery (see above)
- the challenge scoring harness
- estimator bake-off, Hilti dataset handling
