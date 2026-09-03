#!/usr/bin/env bash
# Fast correctness harness: run the first N seconds of a sequence instead of all
# 4951 frames. Full runs take ~15 min, this takes ~1, which is the difference
# between testing an idea and guessing at one.
#
# This checks that the estimator RUNS CORRECTLY -- no crash, lines actually
# tracked rather than churned, residuals sane. It is not a scoring harness:
# 20 s of a 165 s sequence says nothing useful about ATE.
#
#   short_test.sh <config.yaml> <outdir> [secs=20]
set -euo pipefail
W="$(cd "$(dirname "$0")/../../.." && pwd)"
CFG="$1"; OUT="$2"; SECS="${3:-20}"
mkdir -p "$W/$OUT"
# Data/ and SLAM/experiments are absolute symlinks onto the 4TB drive; mount
# the drive at the same path inside the container or they dangle there. The
# mount point moved once already (media -> /mnt/UUID), so RESOLVE it from the
# symlink instead of hardcoding.
DRIVE="$(readlink -f "$W/Data")"; DRIVE="${DRIVE%/INSV_STITCHING/Data}"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -w "/ws/$OUT" -v "$W:/ws" \
  -v "$DRIVE:$DRIVE" \
  -v "$W/SLAM/configs/orbslam3_hilti:/config" insv/orbslam3 bash -c "
  export LD_LIBRARY_PATH=/ws/SLAM/third_party/ORB_SLAM3/lib:/ws/SLAM/third_party/ORB_SLAM3/Thirdparty/DBoW2/lib:/ws/SLAM/third_party/ORB_SLAM3/Thirdparty/g2o/lib:\$LD_LIBRARY_PATH
  stdbuf -oL -eL /ws/SLAM/third_party/ORB_SLAM3/Examples/Monocular-Inertial/mono_rig_euroc \
    /ws/SLAM/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    /ws/$CFG \
    /ws/Data/Hilti/orb/floor_EG_2025-12-02_run_2 \
    /ws/SLAM/configs/orbslam3_hilti/timestamps_run2_first${SECS}s.txt orb > log.txt 2>&1"
echo "-> $OUT/log.txt"
