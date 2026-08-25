# fisheye-slam: build environment for the dual-fisheye SLAM stack.
# Currently: COLMAP (CUDA, sm_120 / Blackwell). OKVIS2 and friends get added here later.
#
# Deps only — source is MOUNTED, never baked in, so patching a camera model is a
# 2-minute ninja rebuild instead of a 1-hour image rebuild. Same pattern as insv/openvins.
#
# Build:  docker build -f docker/fisheye-slam.Dockerfile -t fisheye-slam docker/
#         (context = docker/ — never the repo root, Data/ is hundreds of GB)
#
# Build COLMAP:
#   docker run --rm --gpus all -u $(id -u):$(id -g) -e HOME=/tmp \
#     -v $PWD/SLAM/colmap_src:/src -v $PWD/SLAM/colmap_build:/build \
#     fisheye-slam /usr/local/bin/build_colmap.sh
#
# Run COLMAP:
#   docker run --rm --gpus all -u $(id -u):$(id -g) -e HOME=/tmp \
#     -v $PWD/SLAM/colmap_build:/build -v $PWD/Data:/data \
#     fisheye-slam /build/src/colmap/exe/colmap <args>
#
# NOTE: nothing here ever installs to the host. The system /usr/local/bin/colmap
# is untouched — this image cannot see it.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential pkg-config \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    libcurl4-openssl-dev liblz4-dev libsuitesparse-dev \
    libopenimageio-dev libopenimageio2.4t64 openimageio-tools libopencv-dev \
    libatlas-base-dev libopenblas-openmp-dev libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*

# RTX 5070 Ti is compute capability 12.0 (Blackwell)
ENV CUDAARCHS=120
ENV CMAKE_CUDA_ARCHITECTURES=120

# Build script: configures into /build with the source at /src.
# Deliberately has NO install target — we call the binary in place, so there is
# no code path that could ever write to a host prefix.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'cmake -S /src -B /build -GNinja \' \
    '  -DCMAKE_BUILD_TYPE=Release \' \
    '  -DCMAKE_CUDA_ARCHITECTURES=120 \' \
    '  -DGUI_ENABLED=OFF \' \
    '  -DTESTS_ENABLED=OFF \' \
    '  -DCUDA_ENABLED=ON' \
    'ninja -C /build -j"$(nproc)" colmap_main' \
    'echo "built: /build/src/colmap/exe/colmap"' \
    '/build/src/colmap/exe/colmap --help 2>&1 | head -2' \
    > /usr/local/bin/build_colmap.sh && chmod +x /usr/local/bin/build_colmap.sh

# OKVIS2 build script. Same philosophy: source mounted, no host install.
# USE_NN=OFF: keypoint classification is off — it pulls a multi-GB torch download
# we do not need for the dual-fisheye stack.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'cmake -S /okvis_src -B /okvis_build -GNinja \' \
    '  -DCMAKE_BUILD_TYPE=Release \' \
    '  -DUSE_SYSTEM_CERES=ON \' \
    '  -DBUILD_APPS=ON \' \
    '  -DUSE_NN=OFF \' \
    '  -DBUILD_TESTS=OFF' \
    'ninja -C /okvis_build -j"$(nproc)"' \
    'ls /okvis_build/okvis_apps/ 2>/dev/null | head' \
    > /usr/local/bin/build_okvis.sh && chmod +x /usr/local/bin/build_okvis.sh

WORKDIR /build
CMD ["bash"]
