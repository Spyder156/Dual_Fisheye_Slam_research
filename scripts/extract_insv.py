#!/usr/bin/env python
"""Extract frames + IMU from Insta360 .insv recordings (One RS: _00_ + _10_ pair).

Usage:
    python scripts/extract_insv.py VID_..._00_....insv [VID_..._10_....insv] -o out_dir [--fps 10]

Outputs:
    out_dir/imu.csv           t[s], gx,gy,gz [rad/s], ax,ay,az [m/s^2]
    out_dir/cam0/, cam1/      %06d.png frames per lens (cam0 = _00_ file)
    out_dir/frames.csv        frame index -> timestamp (from container fps)
    out_dir/info.json         probe + IMU stats, for sanity-checking

Units note: telemetry-parser's normalized_imu() reports gyro in deg/s and accel
in g; we convert to rad/s and m/s^2. The printed |a| magnitude should hover
around 9.81 for a mostly-static start — check it on first real data.

Run with the `vision` conda env python (has telemetry_parser installed).
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import telemetry_parser

G = 9.80665


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(video)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = map(int, v["avg_frame_rate"].split("/"))
    return {
        "file": video.name,
        "codec": v["codec_name"],
        "width": v["width"],
        "height": v["height"],
        "fps": num / den,
        "duration_s": float(j["format"]["duration"]),
    }


def extract_imu(video: Path, out_csv: Path) -> dict:
    tp = telemetry_parser.Parser(str(video))
    imu = tp.normalized_imu()
    if not imu:
        return {}
    rows = []
    for s in imu:
        t = s["timestamp_ms"] / 1000.0
        gx, gy, gz = (math.radians(v) for v in s["gyro"])
        ax, ay, az = (v * G for v in s["accl"])
        rows.append((t, gx, gy, gz, ax, ay, az))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gx", "gy", "gz", "ax", "ay", "az"])
        w.writerows(rows)
    dt = (rows[-1][0] - rows[0][0]) / max(len(rows) - 1, 1)
    a_mag = [math.sqrt(r[4]**2 + r[5]**2 + r[6]**2) for r in rows[:500]]
    return {
        "source": video.name,
        "camera": f"{tp.camera} {tp.model}",
        "samples": len(rows),
        "rate_hz": round(1.0 / dt, 1),
        "duration_s": round(rows[-1][0] - rows[0][0], 2),
        "accel_mag_first500_mean": round(sum(a_mag) / len(a_mag), 3),
    }


def extract_frames(video: Path, out_dir: Path, fps: float | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [str(out_dir / "%06d.png")]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path, help="_00_ insv (and optionally _10_)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=None,
                    help="frame extraction rate (default: every frame)")
    ap.add_argument("--no-frames", action="store_true", help="IMU only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info = {"videos": [probe(v) for v in args.videos], "imu": None, "frames": {}}
    for v in info["videos"]:
        print(f"{v['file']}: {v['codec']} {v['width']}x{v['height']} "
              f"@{v['fps']:.2f}fps, {v['duration_s']:.1f}s")

    # IMU: try each file, keep the first that has telemetry
    for v in args.videos:
        stats = extract_imu(v, args.out / "imu.csv")
        if stats:
            info["imu"] = stats
            print(f"IMU from {v.name}: {stats['samples']} samples @ {stats['rate_hz']}Hz, "
                  f"|a| mean {stats['accel_mag_first500_mean']} (expect ~9.81)")
            break
    if not info["imu"]:
        print("WARNING: no IMU telemetry found in any input!", file=sys.stderr)

    if not args.no_frames:
        fps_used = args.fps or info["videos"][0]["fps"]
        with open(args.out / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t"])
            for cam, v in enumerate(args.videos):
                n = extract_frames(v, args.out / f"cam{cam}", args.fps)
                info["frames"][f"cam{cam}"] = n
                print(f"cam{cam}: {n} frames -> {args.out / f'cam{cam}'}")
                if cam == 0:
                    for i in range(n):
                        w.writerow([i + 1, round(i / fps_used, 6)])

    with open(args.out / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"info -> {args.out / 'info.json'}")


if __name__ == "__main__":
    main()
