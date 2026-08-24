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


def _align(pos, n):
    """CDR alignment is relative to the start of the message BODY (after the
    4-byte encapsulation header), not the absolute buffer offset."""
    return (pos + n - 1) & ~(n - 1)


def cdr_string(body, pos):
    """uint32 length (incl. null) + chars. Returns (value, new_pos)."""
    pos = _align(pos, 4)
    (n,) = struct.unpack_from("<I", body, pos)
    pos += 4
    s = body[pos:pos + n - 1].decode("utf-8", "replace")
    return s, pos + n


def parse_header(body, pos):
    """std_msgs/Header: int32 sec, uint32 nanosec, string frame_id."""
    pos = _align(pos, 4)
    sec, nsec = struct.unpack_from("<iI", body, pos)
    pos += 8
    _frame_id, pos = cdr_string(body, pos)
    return sec + nsec * 1e-9, pos


def parse_compressed_image(data):
    """sensor_msgs/CompressedImage: header, string format, uint8[] data."""
    body = data[4:]
    t, pos = parse_header(body, 0)
    _fmt, pos = cdr_string(body, pos)
    pos = _align(pos, 4)
    (n,) = struct.unpack_from("<I", body, pos)
    pos += 4
    return t, body[pos:pos + n]


def parse_imu(data):
    """sensor_msgs/Imu. float64 fields are 8-aligned within the body."""
    body = data[4:]
    t, pos = parse_header(body, 0)
    pos = _align(pos, 8)
    pos += 4 * 8          # orientation (quaternion)
    pos += 9 * 8          # orientation_covariance
    gx, gy, gz = struct.unpack_from("<3d", body, pos)
    pos += 3 * 8
    pos += 9 * 8          # angular_velocity_covariance
    ax, ay, az = struct.unpack_from("<3d", body, pos)
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

    # sanity gate: a correctly parsed accelerometer must average ~9.81 m/s^2
    import math
    mags = [math.sqrt(r[4] ** 2 + r[5] ** 2 + r[6] ** 2) for r in imu_rows[:5000]]
    g_mean = sum(mags) / len(mags)
    if not (8.5 < g_mean < 11.0):
        raise SystemExit(f"IMU PARSE FAILED: mean |a| = {g_mean:.3g}, expected ~9.81. "
                         "CDR offsets are wrong; refusing to write a corrupt dataset.")

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
