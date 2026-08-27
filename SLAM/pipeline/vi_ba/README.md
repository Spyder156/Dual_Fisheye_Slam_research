# vi-ba

Visual-inertial bundle adjustment, lifted out of the Mecka mono-egomotion
pipeline so it can be run standalone here alongside the other SLAM systems.

Takes a COLMAP model plus a trajectory and IMU, and refines the poses against
preintegrated IMU factors. This is the stage that turns a scale-free visual
reconstruction into a metric, gravity-aligned one.

## Layout

```
src/vi_bundle_adjuster.cc   the solver (builds to the `colmap_vi_ba` binary)
docker/Dockerfile.ba        image that builds it against COLMAP + Ceres
lib/                        everything that makes its inputs or reads its outputs
configs/empty_factors.txt   the no-op factor file, used when relpose is unavailable
runner/refine_stage_reference.py
                            the driving logic, extracted verbatim from the
                            pipeline (pipeline.py:325-500) -- NOT importable on
                            its own, kept as the reference for the call order,
                            arguments and failure handling
```

## Order of operations

The solver is one step in the middle; most of the work is preparing what it eats.

1. **`lib/make_imu_factors.py <model_in> <align_json> <prepared> <factors>`**
   Does two things: preintegrates IMU between keyframes into `<factors>`, and
   writes `<prepared>` -- the model rotated into a **gravity-aligned world**,
   with a `world_transform.json` sidecar recording that rotation.
2. **`lib/make_rollpitch_factors.py`** (optional)
   Gravity-derived roll/pitch priors. CoreMotion's heading is not trustworthy
   but its tilt is, which is the whole reason this exists.
3. **`lib/make_relpose_factors.py <raw_traj> <prepared> <rel> [--rollpitch <rp>]`**
   (optional) Relative-pose factors from the front-end trajectory. Falls back to
   `configs/empty_factors.txt` when it cannot be built.
4. **`colmap_vi_ba`** -- the solve:
   ```
   colmap_vi_ba --input_path <prepared> --output_path <out> \
                --factors <rel or empty_factors.txt> \
                --imu_preint <factors> \
                --factor_huber <h> --max_iterations <n>
   ```
   Prints `NO_CONVERGENCE` if it hits the iteration cap. That is **not fatal** --
   the result is still usable, it just did not settle.
5. **`lib/model_to_tum.py ... --unrotate <prepared>/world_transform.json`**
   Undoes step 1's rotation. Skipping this turns an internal convention into a
   global attitude error in the shipped result. The solver does not copy the
   sidecar, so point at the **prepared** model, not the output.
6. **`lib/densify_trajectory.py`** (optional) keyframe rate -> frame rate.

## Sanity checks that matter

`lib/vi_align.py` reports quantities with known-correct answers, and they are
the fastest way to tell whether the input was garbage:

- `|g|` should be ~9.81 m/s^2
- `|b_a|` (accel bias) should be **< 0.5 m/s^2**

A bias near 10 means the solver absorbed a whole gravity vector, i.e. the
trajectory handed to it was degenerate. Seen in practice at `|b_a| = 11.94` with
`scale = 0.0001` on an episode whose VIO had dead-reckoned.

`lib/imu_factor_audit.py` checks the preintegration itself.

## Timeouts

The solver is **silent for the entire solve**, so a stall timer -- not a
progress check -- is what actually bounds it. Budget it against video duration
rather than a flat number; the pipeline uses `clamp(6 x duration, 300, 1800)`
seconds and applies it to both the timeout and the stall.

## Build

```
docker build -f docker/Dockerfile.ba -t vi-ba:latest .
```

Paths inside the Dockerfile assume the pipeline's layout (`/pipeline/lib/...`);
adjust if running the scripts outside the image.
