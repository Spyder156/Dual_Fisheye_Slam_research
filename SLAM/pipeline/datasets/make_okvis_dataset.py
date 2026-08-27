#!/usr/bin/env python
"""Build an EuRoC-style dataset tree that OKVIS2's DatasetReader can consume,
from a folder produced by hilti_bag_to_folder.py.

OKVIS2 expects:
  <out>/imu0/data.csv          ts_ns, wx, wy, wz, ax, ay, az
  <out>/cam0/data.csv          ts_ns, filename
  <out>/cam0/data/<file>.png   (or .jpg)
  <out>/cam1/...

Images are symlinked, not copied — the source frames stay the single copy.

Usage: make_okvis_dataset.py <src_ds_dir> <out_dir>
"""
import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    frames = np.genfromtxt(args.src / "frames.csv", delimiter=",", names=True)
    fidx = np.atleast_1d(frames["frame"]).astype(int)
    ft = np.atleast_1d(frames["t"])

    for cam in ("cam0", "cam1"):
        d = args.out / cam / "data"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, t in zip(fidx, ft):
            src = (args.src / cam / f"{i:06d}.jpg").resolve()
            if not src.exists():
                continue
            ts_ns = int(round(t * 1e9))
            name = f"{ts_ns}.jpg"
            link = d / name
            if not link.exists():
                # relative symlink: must resolve inside the container too, where
                # the repo is mounted at a different prefix than on the host
                os.symlink(os.path.relpath(src, d.resolve()), link)
            rows.append((ts_ns, name))
        with open(args.out / cam / "data.csv", "w") as f:
            f.write("#timestamp [ns],filename\n")
            for ts, n in rows:
                f.write(f"{ts},{n}\n")
        print(f"{cam}: {len(rows)} frames")

    imu = np.genfromtxt(args.src / "imu.csv", delimiter=",", names=True)
    (args.out / "imu0").mkdir(parents=True, exist_ok=True)
    with open(args.out / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for r in zip(imu["t"], imu["gx"], imu["gy"], imu["gz"],
                     imu["ax"], imu["ay"], imu["az"]):
            f.write(f"{int(round(r[0]*1e9))},{r[1]:.9f},{r[2]:.9f},{r[3]:.9f},"
                    f"{r[4]:.9f},{r[5]:.9f},{r[6]:.9f}\n")
    print(f"imu0: {len(imu['t'])} samples")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
