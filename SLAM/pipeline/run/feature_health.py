#!/usr/bin/env python
"""Feature-health metrics from a keypoint dump.

WHY NOT ATE: a front-end change moves ATE by less than the run-to-run noise
floor (0.79 score on ORB-SLAM3), and one catastrophic blackout dominates the
score anyway. These metrics are per-frame, so a front-end change is measurable
in ONE run.

Reads kp_*.csv (t,cam,id,u,v,tracked) and reports:
  - % of frames with fewer than THRESH tracked features (both cameras)
  - the longest contiguous blackout, in seconds
  - tracked features per frame, per camera

The failure we are chasing on floor_EG run_2 is a 1.0 s blackout ~8 s before the
end, where both lenses drop to 0-2 tracked features (dark + motion blur), after
which re-acquisition puts the pose 43 m away.
"""
import argparse
from pathlib import Path

import numpy as np


def load(path, max_rows=None):
    """t -> (tracked_cam0, tracked_cam1). Streamed: these files reach 1 GB."""
    cnt = {}
    with open(path) as fh:
        fh.readline()
        for i, l in enumerate(fh):
            if max_rows and i > max_rows:
                break
            p = l.split(",")
            try:
                t = round(float(p[0]), 3); c = int(p[1]); tr = int(p[5])
            except (ValueError, IndexError):
                continue
            k = (t, c)
            cnt[k] = cnt.get(k, 0) + tr
    ts = sorted({k[0] for k in cnt})
    return np.array([[t, cnt.get((t, 0), 0), cnt.get((t, 1), 0)] for t in ts])


def report(a, thresh, label):
    tot = a[:, 1] + a[:, 2]
    dead = tot < thresh
    runs, s = [], None
    for i, d in enumerate(dead):
        if d and s is None:
            s = i
        if not d and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(dead)))
    runs = [r for r in runs if r[1] - r[0] >= 3]
    longest = max(((a[e-1, 0] - a[s_, 0]) for s_, e in runs), default=0.0) if runs else 0.0
    print(f"{label:28s} frames {len(a):5d}  "
          f"dead {100*dead.mean():5.2f}%  longest {longest:5.2f}s  "
          f"med/frame cam0 {np.median(a[:,1]):6.0f} cam1 {np.median(a[:,2]):6.0f}")
    return dict(dead_pct=100*dead.mean(), longest=longest,
                med0=float(np.median(a[:, 1])), med1=float(np.median(a[:, 2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="kp_*.csv files (or dirs holding one)")
    ap.add_argument("--thresh", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    for d in a.dumps:
        p = Path(d)
        if p.is_dir():
            hits = list(p.glob("kp_*.csv"))
            if not hits:
                print(f"{d}: no kp_*.csv"); continue
            p = hits[0]
        report(load(p, a.max_rows), a.thresh, p.parent.name)


if __name__ == "__main__":
    main()
