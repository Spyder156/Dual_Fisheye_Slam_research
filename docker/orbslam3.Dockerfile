# ORB-SLAM3 baseline image (no ROS — we drive it with the native examples).
# ORB-SLAM3 wants OpenCV >= 4.4, so we build a minimal OpenCV 4.5.4 (core
# modules only) rather than using focal's 4.2.
#
# Build:  docker build -f docker/orbslam3.Dockerfile -t insv/orbslam3 docker/
#         (context = docker/ — never the repo root, Data/ is 78G)
# Run:    docker run -it --rm -v $PWD/SLAM/ORB_SLAM3:/ORB_SLAM3 \
#             -v $PWD/Data:/data insv/orbslam3
# Inside: cd /ORB_SLAM3 && ./build.sh
FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip pkg-config \
    libeigen3-dev libboost-serialization-dev libssl-dev \
    libgl1-mesa-dev libglew-dev libepoxy-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Minimal OpenCV 4.5.4
RUN cd /tmp && wget -q https://github.com/opencv/opencv/archive/4.5.4.zip \
    && unzip -q 4.5.4.zip && cd opencv-4.5.4 && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIST=core,imgproc,imgcodecs,videoio,features2d,calib3d,highgui \
       -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF .. \
    && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/opencv-4.5.4 /tmp/4.5.4.zip

# Pangolin (v0.8 builds cleanly on 20.04)
RUN cd /tmp && git clone --depth 1 --branch v0.8 https://github.com/stevenlovegrove/Pangolin.git \
    && cd Pangolin && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF .. \
    && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/Pangolin

CMD ["bash"]
