# Run Output Contract

**Every SLAM run produces one folder.** No exceptions, no ad-hoc outputs. If a run doesn't produce this, it isn't finished.

```
SLAM/experiments/<run_name>/
├── run.rrd                 # the Rerun recording (spec below)
├── traj.csv                # trajectory, t,px,py,pz,qx,qy,qz,qw
├── points.csv              # SLAM landmark cloud, x,y,z
├── stats.csv               # per-frame reprojection rms + feature counts
├── config/                 # exact config used, copied verbatim
├── score.txt               # ATE RMSE, challenge score, coverage vs GT
└── viz/                    # still images (see §2)
```

---

## 1. `run.rrd` — the Rerun recording

**Layout: 3D world pane is the LARGEST.** Everything runs at full frame rate.

### Main pane — 3D world (biggest)
- **SLAM camera path**, growing over time
- **SLAM point cloud** — the landmarks the SLAM actually built.
  **Coloured with the real scene appearance**, never a false-colour ramp: each
  landmark is projected into nearby camera poses with the actual calibrated
  model (KB4) and the pixel is sampled. Landmarks that never project into a
  frame stay neutral grey, and the coloured fraction is reported.
  A height/turbo ramp is *forbidden* — it invents structure that isn't measured
  and makes a map impossible to compare against the photograph it came from.
  On monochrome sensors (Hilti is `camera_type: gray`) this yields grey levels;
  say so rather than implying RGB. Radius `0.00025·S`.
- **COLMAP point cloud** when a refinement pass exists — *same pane, same world frame*, so the two are directly comparable (not for now)
- GT path overlaid when ground truth exists, aligned into the same frame
- **Live camera frusta — one real `rr.Pinhole` per physical camera**, placed at
  `R_CtoG = R_ItoG·R_CtoI`, `p_CinG = p_IinG + R_ItoG·p_CinI`, extrinsics read
  from the run's own config. Never a dot for "the camera": a frustum shows
  where each lens points, which for a back-to-back rig is the whole question.
  Sized `image_plane_distance = 0.005·S` — a camera should look like a camera
  (~15 cm at room scale), not like a feature of the map.
  *Caveat:* the frustum is the equivalent-pinhole cone from `(fx,fy,cx,cy)`,
  ≈115° — it does **not** depict the true ~195° fisheye field.

### Everything scales with `S`
`S = |bbox diagonal of the trajectory|`. **No visual size is ever a hardcoded
metre value.** Points `0.00025·S`, path `0.0025·S`, frustum `0.005·S`, eye
`~1.3·S`. A viz tuned by eye for one room silently breaks on the next dataset.

### Side pane — video
- The actual camera frames, full fps
- **Upright for a human.** Detect it, don't guess: take mean accelerometer =
  world-up in the IMU frame, map it through `R_CtoI` into the camera. If it
  lands on image **+y (down)**, the sensor is mounted upside down — rotate the
  display 180° and rotate the keypoint coordinates with it.
  This is **display only**: the calibration already encodes the mounting, so
  SLAM must keep consuming the raw frames. Print which rotation was applied so
  a genuine convention bug is never hidden behind a cosmetic flip.
  *(Hilti floor_EG: world-up sits at +0.97 on image +y — 172° from upright.)*
- **Current-frame keypoints** overlaid, bright
- **Tracked keypoints persist**: a matched feature keeps its history drawn, each older observation **progressively lighter**, so a track reads as a fading tail. This is how we see tracking quality directly rather than inferring it.
- **Masked regions are shaded, never blanked** — darkened to 38 % with a red
  tint. The masked area stays readable as context, so we can see *what* is being
  thrown away and judge whether the mask is cutting too much. Blanking hides
  exactly the evidence needed to tell a good mask from a bad one.
  If a run has no masks, say so in the report rather than silently showing none.

### Below the video — two running plots
- **Reprojection error** (px, per frame)
- **Rotation error** — gyro-integrated rotation vs optical/estimated rotation

### Scale — the view must be readable the moment it opens
Rerun auto-fits the 3D view to **all** logged data, so a few near-infinity
landmarks shrink the whole trajectory to a dot. Every run must therefore:

- **Clip the displayed cloud** to `1.5 × trajectory diagonal` around the path.
  Print how many landmarks were dropped — never clip silently. `points.csv`
  still stores the **full** map; clipping is a display concern only.
- **Derive every visual size from scene scale** `S = |bbox diagonal of the path|`:
  landmark radius `0.0012·S`, path width `0.0025·S`, current-pose dot `0.010·S`.
  Never hardcode metre sizes — they only look right for one room size.
- **Place the eye explicitly** (`EyeControls3D`): 3/4 view from above at
  `~1.3·S`, looking at the path centre, `eye_up = +Z`. Do not rely on auto-fit.
- **1 m `LineGrid3D`** on the ground plane, so distances are readable by eye.
- Log `rr.ViewCoordinates.RIGHT_HAND_Z_UP` on `/world`.

### Hard rules
- Full frame rate. No decimation unless stated.
- **Never `static=True`** — it hangs Rerun viewer 0.33 on images. Everything on the timeline.
- Verify with `rerun rrd verify <file>` before reporting the run as done.

---

## 2. `viz/` — still images

**"Visualization" means showing what the system sees. Not plots, not histograms.**

- `matches_*.png` — keypoint matches drawn between frame pairs, several points in the sequence
- `path_topdown.png` — camera path viewed from above, with GT if available
- Anything else that shows the system's actual perception and is genuinely informative

Matplotlib plots are supporting material at best. They are never the deliverable.

---

## 3. Reporting

Results and discussion go **in chat**, as an experiment card: input → processing → output → tests → interpretation last. This file defines artifacts only.
