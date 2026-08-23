#!/usr/bin/env python
"""Convert a Hilti-Trimble ROS2 rosbag (.db3) to the run_folder dataset layout.

Reads the sqlite3 bag directly — no ROS install needed. CompressedImage and Imu
messages are CDR-encoded; we parse just the fields we need.

Outputs:
  <out>/cam0/%06d.jpg, cam1/%06d.jpg   (frame index = 1-based, shared timeline)
  <out>/frames.csv                     frame,t
  <out>/imu.csv                        t,gx,gy,gz,ax,ay,az
  <out>/info.json

Usage: hilti_bag_to_folder.py <rosbag.db3> <out_dir> [--max-frames N]
"""
import argparse
import json
import sqlite3
import struct
from pathlib import Path


def cdr_string(buf, off):
    """Read a CDR-encoded string; returns (value, new_offset)."""
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    s = buf[off:off + n - 1].decode("utf-8", "replace")
    off += n
    off = (off + 3) & ~3  # align 4
    return s, off


def parse_header(buf, off):
    """std_msgs/Header: stamp(sec int32, nsec uint32) + frame_id string."""
    sec, nsec = struct.unpack_from("<iI", buf, off)
    off += 8
    _frame_id, off = cdr_string(buf, off)
    return sec + nsec * 1e-9, off


def parse_compressed_image(data):
    """sensor_msgs/CompressedImage: header, format string, uint8[] data."""
    off = 4  # CDR encapsulation header
    t, off = parse_header(data, off)
    _fmt, off = cdr_string(data, off)
    (n,) = struct.unpack_from("<I", data, off)
    off += 4
    return t, data[off:off + n]


def parse_imu(data):
    """sensor_msgs/Imu: header, orientation(4d)+cov(9d), ang_vel(3d)+cov(9d),
    lin_acc(3d)+cov(9d). Doubles are 8-aligned."""
    off = 4
    t, off = parse_header(data, off)
    off = (off + 7) & ~7
    off += 4 * 8 + 9 * 8          # orientation + covariance
    gx, gy, gz = struct.unpack_from("<3d", data, off)
    off += 3 * 8 + 9 * 8          # angular velocity + covariance
    ax, ay, az = struct.unpack_from("<3d", data, off)
    return t, (gx, gy, gz, ax, ay, az)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cam0").mkdir(exist_ok=True)
    (args.out / "cam1").mkdir(exist_ok=True)

    con = sqlite3.connect(str(args.bag))
    topics = {name: tid for tid, name in
              con.execute("select id, name from topics").fetchall()}

    # IMU
    imu_rows = []
    tid = topics["/imu/data_raw"]
    for (blob,) in con.execute(
            "select data from messages where topic_id=? order by timestamp", (tid,)):
        t, v = parse_imu(blob)
        imu_rows.append((t, *v))
    imu_rows.sort()

    # images, keyed by timestamp so the two cameras share a frame index
    per_cam = {}
    for cam in ("cam0", "cam1"):
        tid = topics[f"/{cam}/image_raw/compressed"]
        rows = []
        for (blob,) in con.execute(
                "select data from messages where topic_id=? order by timestamp", (tid,)):
            rows.append(parse_compressed_image(blob))
        rows.sort(key=lambda r: r[0])
        per_cam[cam] = rows
    con.close()

    n = min(len(per_cam["cam0"]), len(per_cam["cam1"]))
    if args.max_frames:
        n = min(n, args.max_frames)

    # timestamps: use cam0 as the shared timeline (they are hardware-synced)
    t0_img = per_cam["cam0"][0][0]
    dt = [abs(per_cam["cam0"][i][0] - per_cam["cam1"][i][0]) for i in range(min(n, 500))]
    max_skew_ms = max(dt) * 1e3 if dt else 0.0

    with open(args.out / "frames.csv", "w") as f:
        f.write("frame,t\n")
        for i in range(n):
            t = per_cam["cam0"][i][0]
            f.write(f"{i+1},{t:.9f}\n")
            for cam in ("cam0", "cam1"):
                (args.out / cam / f"{i+1:06d}.jpg").write_bytes(per_cam[cam][i][1])

    with open(args.out / "imu.csv", "w") as f:
        f.write("t,gx,gy,gz,ax,ay,az\n")
        for r in imu_rows:
            f.write(",".join(f"{x:.9f}" for x in r) + "\n")

    imu_rate = (len(imu_rows) - 1) / (imu_rows[-1][0] - imu_rows[0][0])
    cam_rate = (n - 1) / (per_cam["cam0"][n - 1][0] - t0_img)
    info = {
        "bag": str(args.bag),
        "frames": n,
        "imu_samples": len(imu_rows),
        "imu_rate_hz": round(imu_rate, 2),
        "cam_rate_hz": round(cam_rate, 2),
        "cam0_cam1_max_skew_ms": round(max_skew_ms, 4),
        "t_img_first": per_cam["cam0"][0][0],
        "t_imu_first": imu_rows[0][0],
        "t_imu_last": imu_rows[-1][0],
    }
    (args.out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
