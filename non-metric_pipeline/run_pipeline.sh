#!/usr/bin/env bash
# End-to-end: a dual-fisheye clip -> per-unit calibration -> SLAM -> refinement.
#
#   run_pipeline.sh <dataset_dir> <out_dir> [--skip-calib]
#
# dataset_dir must contain:
#   cam0/%06d.jpg  cam1/%06d.jpg  frames.csv (frame,t)  imu.csv (t,gx..az)
# imu.csv is optional -- it is only used if you later want a metric anchor.
#
# The stages are deliberately separate: stage 0 is per-DEVICE and only needs
# running again when the hardware changes, not per clip.
set -euo pipefail

DS="${1:?dataset dir}"; OUT="${2:?output dir}"; shift 2
SKIP_CALIB=0
for a in "$@"; do [ "$a" = "--skip-calib" ] && SKIP_CALIB=1; done

R="$(cd "$(dirname "$0")/.." && pwd)"
P="$R/SLAM/pipeline"
mkdir -p "$OUT"

# ---- 0. per-unit calibration ------------------------------------------------
CAL="$OUT/calib"
if [ "$SKIP_CALIB" = "0" ]; then
  echo "== stage 0: per-unit calibration from the clip itself"
  python "$P/step0_calib/step0_rig_calib.py" \
      --dataset "$DS" --out "$CAL" \
      --stride 5 --window 8 --max-frames 500 --offsets 3,6,12,24,48 --gpu \
      --colmap "$P/vi_ba_kb/lib/colmap_docker.sh"
  python "$P/step0_calib/step0_extract_rig.py" "$CAL/sparse" | tee "$OUT/rig_report.txt"
else
  echo "== stage 0 skipped (reusing $CAL)"
fi

# ---- 1. SLAM ----------------------------------------------------------------
echo "== stage 1: SLAM, both lenses, rig extrinsics fixed"
echo "   (config must carry Rig.T_c0_c1 from stage 0 and Rig.baseline_from_start: 1)"
echo "   a ZERO baseline here is far worse than no rig -- 5.3 vs 84.7 on our reference clip"

# ---- 2. refinement ----------------------------------------------------------
echo "== stage 2: refinement"
echo "   only worth running with structure the front end did not already imply"
echo "   (loop closures, or an independent reconstruction)"

echo "done -> $OUT"
