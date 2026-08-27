#!/usr/bin/env bash
# COLMAP shim: runs the containerised binary but mounts the repo at its OWN
# host path, so paths are identical inside and outside and the caller needs no
# path translation.
R=/home/raghav/workspace/INSV_STITCHING
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e LD_LIBRARY_PATH=$R/SLAM/build/colmap_build/_deps/onnxruntime-build/lib \
  --gpus all -v "$R:$R" -w "$PWD" \
  fisheye-slam "$R/SLAM/build/colmap_build/src/colmap/exe/colmap" "$@"
