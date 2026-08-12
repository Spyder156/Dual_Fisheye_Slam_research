# Dual Fisheye SLAM Research

IMU-based SLAM for dual-fisheye 360 cameras (Insta360 One RS / X-series): two ~200° lenses back-to-back, one IMU, no usable stereo overlap — i.e. **multi-camera monocular-rig VIO** with online rig refinement and a >180°-valid camera model (Mei / omni-radtan).

## Layout

- `SLAM/SLAMS.md` — candidate survey, camera-model decision, roadmap. Start here.
- `SLAM/<repo>/` — cloned SLAM bases (gitignored; clone them yourself, see SLAMS.md).
- `scripts/` — calibration fitting, stitching, pose comparison, visualization tools.
- `Markdowns/` — project notes and ideas.
- `Data/` (gitignored) — FIORD benchmark (photo-mode, lidar GT, **no IMU**) + own recordings.
- `MediaSDK/` (gitignored) — proprietary Insta360 SDK.

## Status

Survey + benchmark phase. FIORD vision-only arms done (see SLAMS.md); VIO track waiting on own One RS recordings + Kalibr omni-radtan calibration.
