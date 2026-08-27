# SLAM

Dual-fisheye visual-inertial SLAM for Insta360-class rigs.
Target: Hilti x Trimble SLAM Challenge 2026.

## Layout

| dir | what |
|---|---|
| `third_party/` | upstream repos, pinned. **Never edit directly** — see `patches/` |
| `patches/`     | our diffs against `third_party/`, plus files we added |
| `build/`       | build trees (gitignored) |
| `pipeline/`    | our code, by stage |
| `configs/`     | per-dataset, per-estimator configs |
| `experiments/` | one folder per run, per the `docs/OUTPUT.md` contract |
| `docs/`        | OUTPUT.md (run contract), SOLUTIONS.md, experiments.md, SLAMS.md |

## pipeline/

| stage | what |
|---|---|
| `datasets/`    | .insv / rosbag -> our folder layout, EuRoC converters |
| `step0_calib/` | per-unit intrinsics + rig extrinsics + IMU metric scale, from images alone |
| `vi_ba/`       | visual-inertial bundle adjustment (Ceres): IMU preint + relpose + roll/pitch factors |
| `run/`         | estimator runners, bake-off, scoring |
| `viz/`         | run-output builder (`docs/OUTPUT.md` contract) |
| `okvis_probe/` | convention/geometry probe for the OKVIS family |

## Rules
- Every run produces a folder per `docs/OUTPUT.md`. No ad-hoc outputs.
- Upstream repos are read-only; changes live in `patches/`.
- Data lives in `Data/` at the repo root, gitignored.
