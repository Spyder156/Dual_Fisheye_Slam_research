#!/usr/bin/env python
"""Finite-extent reconstruction, validated OFFLINE on a saved run.

For every verified multi-KF line: fit the infinite line from ALL saved
observations (poses fixed), then recover the VISIBLE intervals:
  - P0 = d x m (point on line closest to origin), line = P0 + s*d
  - each observed endpoint ray is intersected with the line -> scalar s,
    weighted by the ray/line angle (shallow rays clipped, not the whole obs)
  - intervals supported by >= 2 distinct keyframes are fused; gaps preserved
  - endpoints stored EXACTLY on the fitted line (P0 + s*d, by construction)

Outputs:
  <out>/rebuilt.csv    lineid,seg,x1,y1,z1,x2,y2,z2,nkf,s0,s1
  <out>/census.csv     lineid,nkf,resid_deg,old_len,new_len,reason
  stdout               the 171367-style before/after + missing-extent census

usage: rebuild_extents.py <rundir> --config <yaml> --out DIR [--min-sin 0.1]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from debug_line_projection import load_cam, unproject_kb4  # noqa: E402


def fit_line(planes):
    """planes: list of (n_w unit, c_w camera centre). Returns (P0, d) or None.
    d: common null direction of the normals; P0: LS point with gauge P0.d=0."""
    N = np.zeros((3, 3))
    for n, _ in planes:
        N += np.outer(n, n)
    w, V = np.linalg.eigh(N)
    if w[1] < 5 * w[0] + 1e-9:
        return None, "direction_unobservable"
    d = V[:, 0]
    A = np.outer(d, d)
    b = np.zeros(3)
    for n, c in planes:
        A += np.outer(n, n)
        b += n * (n @ c)
    if np.linalg.eigvalsh(A)[0] < 1e-6:
        return None, "point_unobservable"
    P0 = np.linalg.solve(A, b)
    P0 = P0 - d * (P0 @ d)   # gauge: closest point to origin along d
    return (P0, d), None


def ray_line_s(P0, d, c, b):
    """s on line P0+s*d of the closest point to ray c+t*b.
    Returns (s, t, sin_angle)."""
    db = d @ b
    den = 1.0 - db * db
    u = P0 - c
    if den < 1e-9:
        return None
    s = (-(u @ d) + db * (u @ b)) / den
    t = db * s + (u @ b)
    return s, t, np.sqrt(max(0.0, den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-sin", type=float, default=0.1,
                    help="reliability floor on the ray/line angle (~5.7 deg); "
                         "an end below it is CLIPPED, the rest of the obs kept")
    ap.add_argument("--max-depth", type=float, default=60.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cams = [load_cam(args.config, c) for c in (0, 1)]
    obs = np.genfromtxt(args.rundir / "ml_orb_kfobs.csv", delimiter=",", names=True)
    pos = np.genfromtxt(args.rundir / "ml_orb_kfpose_b0.csv", delimiter=",", names=True)
    P = {}
    for r in pos:
        R = np.array([[r["r00"], r["r01"], r["r02"]],
                      [r["r10"], r["r11"], r["r12"]],
                      [r["r20"], r["r21"], r["r22"]]])
        P[(int(r["kfid"]), int(r["cam"]))] = (R, np.array([r["tx"], r["ty"], r["tz"]]))

    # exported (old) extents + the ESTIMATOR'S verified geometry. Where the
    # estimator already has a validated line, extent is rebuilt ON that line
    # (reprojection quality unchanged by construction); the offline eigen-fit
    # is only for lines that never got a finite extent, so have no export.
    old_len, stored = {}, {}
    try:
        ml = np.atleast_1d(np.genfromtxt(args.rundir / "ml_orb.csv", delimiter=",", names=True))
        for r in ml:
            e1 = np.array([r["x1"], r["y1"], r["z1"]]); e2 = np.array([r["x2"], r["y2"], r["z2"]])
            L = float(np.linalg.norm(e2 - e1))
            old_len[int(r["id"])] = L
            if L > 1e-6:
                d = (e2 - e1) / L
                stored[int(r["id"])] = (e1 - d * (e1 @ d), d)   # gauge P0.d=0
    except OSError:
        pass

    by_line = defaultdict(list)
    for r in np.atleast_1d(obs):
        by_line[int(r["lineid"])].append(r)

    rebuilt = open(args.out / "rebuilt.csv", "w")
    rebuilt.write("lineid,seg,x1,y1,z1,x2,y2,z2,nkf,s0,s1\n")
    census = open(args.out / "census.csv", "w")
    census.write("lineid,nkf,resid_deg,old_len,new_len,reason\n")

    reasons = defaultdict(int)
    lens_old, lens_new = [], []
    n_ok = 0
    for lid, rows in sorted(by_line.items()):
        kfs = {int(r["kfid"]) for r in rows}
        if len(kfs) < 2:
            continue
        planes, frags = [], []
        for r in rows:
            key = (int(r["kfid"]), int(r["cam"]))
            if key not in P:
                continue
            R, t = P[key]
            c_w = -R.T @ t
            b = unproject_kb4(np.array([r["x1"], r["x2"]]),
                              np.array([r["y1"], r["y2"]]), cams[int(r["cam"])])
            b_w = (R.T @ b.T).T
            n_w = np.cross(b_w[0], b_w[1])
            nn = np.linalg.norm(n_w)
            if nn < 1e-9:
                continue
            planes.append((n_w / nn, c_w))
            frags.append((int(r["kfid"]), c_w, b_w))
        if len(planes) < 2:
            reasons["too_few_usable_obs"] += 1
            census.write(f"{lid},{len(kfs)},nan,{old_len.get(lid, 0):.3f},0,too_few_usable_obs\n")
            continue
        if lid in stored:
            P0, d = stored[lid]                 # estimator's verified line
        else:
            fit, why = fit_line(planes)
            if fit is None:
                reasons[why] += 1
                census.write(f"{lid},{len(kfs)},nan,{old_len.get(lid, 0):.3f},0,{why}\n")
                continue
            P0, d = fit
        # reprojection quality of the FIT (must stay as good as the audit).
        # A consistent observation has its plane normal PARALLEL to the line's
        # moment about that camera (both are the plane's normal), so the
        # residual is the angle BETWEEN them: arccos, not arcsin.
        worst = 0.0
        for n, c in planes:
            m_c = np.cross(P0 - c, d)
            nm = np.linalg.norm(m_c)
            if nm < 1e-9:
                worst = 180.0; break
            worst = max(worst, np.degrees(np.arccos(min(1.0, abs(n @ (m_c / nm)) ))))
        # visible intervals: per fragment, s of each endpoint ray on the line;
        # clip UNRELIABLE ENDS (shallow angle / behind / absurd depth), keep
        # the rest of the fragment instead of rejecting it whole
        ivals = []   # (s_lo, s_hi, kfid)
        for kfid, c_w, b_w in frags:
            ss = []
            for b in b_w:
                r = ray_line_s(P0, d, c_w, b)
                if r is None:
                    continue
                s, t, sin_a = r
                if sin_a < args.min_sin or t <= 0.05 or t > args.max_depth:
                    continue          # clip THIS end only
                ss.append(s)
            if len(ss) == 2 and abs(ss[1] - ss[0]) > 1e-4:
                ivals.append((min(ss), max(ss), kfid))
        if not ivals:
            reasons["no_reliable_interval"] += 1
            census.write(f"{lid},{len(kfs)},{worst:.3f},{old_len.get(lid, 0):.3f},0,no_reliable_interval\n")
            continue
        # fuse: sweep the union, keep sub-intervals covered by >= 2 distinct
        # KFs (single-KF cover only counts if that is all there is: >=2 KFs
        # total ensures the LINE is multiview; extent portions need 2 too),
        # preserve gaps
        edges = sorted({s for iv in ivals for s in iv[:2]})
        segs = []
        minKF = 2 if len({iv[2] for iv in ivals}) >= 2 else 1
        for a, b2 in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (a + b2)
            cover = {kf for lo, hi, kf in ivals if lo <= mid <= hi}
            if len(cover) >= minKF:
                if segs and abs(segs[-1][1] - a) < 1e-6 and segs[-1][2] == len(cover) >= 0:
                    segs[-1] = (segs[-1][0], b2, max(segs[-1][2], len(cover)))
                else:
                    segs.append((a, b2, len(cover)))
        # merge touching
        fused = []
        for s0, s1, nkf in segs:
            if fused and abs(fused[-1][1] - s0) < 1e-6:
                fused[-1] = (fused[-1][0], s1, max(fused[-1][2], nkf))
            else:
                fused.append((s0, s1, nkf))
        fused = [f for f in fused if f[1] - f[0] > 0.02]
        if not fused:
            reasons["coverage_below_2kf"] += 1
            census.write(f"{lid},{len(kfs)},{worst:.3f},{old_len.get(lid, 0):.3f},0,coverage_below_2kf\n")
            continue
        tot = sum(s1 - s0 for s0, s1, _ in fused)
        n_ok += 1
        lens_new.append(tot)
        if lid in old_len:
            lens_old.append(old_len[lid])
        for k, (s0, s1, nkf) in enumerate(fused):
            e1, e2 = P0 + s0 * d, P0 + s1 * d
            rebuilt.write(f"{lid},{k},{e1[0]:.4f},{e1[1]:.4f},{e1[2]:.4f},"
                          f"{e2[0]:.4f},{e2[1]:.4f},{e2[2]:.4f},{nkf},{s0:.4f},{s1:.4f}\n")
        census.write(f"{lid},{len(kfs)},{worst:.3f},{old_len.get(lid, 0):.3f},{tot:.3f},ok\n")
        if lid == 171367:
            print(f"line 171367: {len(kfs)} KFs, fit worst resid {worst:.2f} deg, "
                  f"old extent {old_len.get(lid, 0)*100:.0f} cm -> rebuilt "
                  f"{tot*100:.0f} cm in {len(fused)} interval(s)")

    lens_new = np.array(lens_new); lens_old = np.array(lens_old)
    print(f"\nlines with >=2 KFs: {len(by_line)} -> rebuilt extents for {n_ok}")
    print("missing/failed extents by reason:",
          dict(sorted(reasons.items(), key=lambda kv: -kv[1])))
    if len(lens_old):
        print(f"exported-before lengths: median {np.median(lens_old)*100:.0f} cm")
    if len(lens_new):
        print(f"rebuilt lengths:        median {np.median(lens_new)*100:.0f} cm, "
              f"p90 {np.percentile(lens_new, 90)*100:.0f} cm, "
              f">=50cm: {(lens_new >= 0.5).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
