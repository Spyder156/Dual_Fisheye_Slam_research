# Experiment B — zero-baseline rig, refined online

## The issue

`Rig.T_c0_c1` carries a **metric** translation: 4.01 cm, measured by Step 0
(COLMAP rig BA + IMU scale, validated against Hilti's 4.01 cm truth).

ORB-SLAM3's monocular map is in **arbitrary units** until the IMU initialisation
converges, and even then the scale drifts. Measured across our three sequences:

| sequence | first 10% | first 25% | last 25% | whole |
|---|---|---|---|---|
| 12-02_run_1 | 0.988 | 0.985 | 0.980 | 0.976 |
| 12-02_run_2 | 0.993 | 0.992 | **1.683** | 1.142 |
| 10-16_run_1 | 1.048 | 1.018 | 1.002 | 1.004 |

So inserting "4.01 cm" into the map places the two cameras **4.01 map units**
apart, which is 4.01 cm only by accident. If the map scale is such that
4 cm reads as 1 m, the rig starts a metre wide and the two cameras never
reconcile -- visible in the run.rrd as two frusta that track together
perfectly but sit at an obviously wrong separation.

A single global scale correction afterwards cannot fix it: by then the
inconsistency is baked into the map.

## Rejected alternatives

**Express the baseline in current map units.** Requires knowing the scale, which
means aligning after the fact -- at which point the two cameras were never
working together *online*. That kills cross-lens loop closure, which is the one
capability a back-to-back rig has and a single camera structurally cannot have.

**Wait for IMU convergence, then apply the metric extrinsic.** The IMU takes
time to converge, so the beginning of every sequence runs without the rig.
Kept as **Experiment A**, as a control.

## The solution (Experiment B)

Start the rig with a **zero baseline** and let it refine itself, the way COLMAP
refines rig extrinsics during its own solve.

Why zero works:

- **A zero baseline is scale-invariant.** 0 map units = 0 metres at any scale.
  There is no unit mismatch to get wrong.
- **Rotation is also scale-free**, so the valuable part of the rig -- rotation
  observability from a 180-degree feature spread -- is correct from frame 1.
- At 4 cm against metre-scale depth, treating the rig as central costs well
  under a pixel of reprojection error.
- Both cameras feed one map from the start, so pooled-BoW cross-lens loop
  closure is available immediately.

### The risk this design has to manage

In a monocular map, **map scale and rig baseline are degenerate**: scaling the
map up while shrinking the baseline gives nearly identical reprojections. Our
own Step 0 measured the rig translation as weakly observable on a
non-overlapping rig (+/-30% spread before pooling; rig BA *degraded* it by 24%
when left free). A fully free baseline will wander.

So the baseline is released in two phases:

| phase | condition | baseline |
|---|---|---|
| 1 | before IMU scale has converged | **fixed at 0** |
| 2 | after | released, with a **soft prior** toward `4.01 cm / map_scale` |

Never fully free.

### Success criterion (falsifiable)

The refined baseline is logged every N keyframes, converted to metres through
the live map scale. Success = it converges toward **4.01 cm** and stays there.
If it wanders to 15 cm or 0.5 cm, the degeneracy won and Experiment A's number
tells us what the ceiling was.

## Rotation is unaffected throughout

Rotation carries no scale, so `Rig.refine_rotation` stays on in both phases.
That is also the part our data says is well observed (Step 0: angle recovered to
0.03 deg, baseline only to ~1% and only after pooling).
