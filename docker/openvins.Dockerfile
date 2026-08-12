# Deps-only image for OpenVINS (ROS1 Noetic). Source is mounted, not baked in,
# so we can iterate on the CamMei port without rebuilding the image.
#
# Build:  docker build -f docker/openvins.Dockerfile -t insv/openvins docker/
#         (context = docker/ — never the repo root, Data/ is 78G)
# Run:    docker run -it --rm -v $PWD/SLAM/open_vins:/ws/src/open_vins \
#             -v $PWD/Data:/data insv/openvins
# Inside: cd /ws && catkin build && source devel/setup.bash
FROM ros:noetic-perception

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libeigen3-dev libboost-all-dev libceres-dev \
    python3-catkin-tools python3-osrf-pycommon \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /ws/src && cd /ws && catkin config --init --extend /opt/ros/noetic \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

WORKDIR /ws
CMD ["bash"]
