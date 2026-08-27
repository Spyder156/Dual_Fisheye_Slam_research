#!/usr/bin/env bash
# Estimator bake-off: run every built estimator on ONE sequence, produce the
# standard run folder (SLAM/docs/OUTPUT.md) for each, and print a comparison table.
#
#   scripts/run_bakeoff.sh [sequence_name]
#
# Estimators that are not built are SKIPPED with a loud message -- never
# silently omitted, so the summary can't look complete when it isn't.
#
# Safe to leave running unattended: each estimator is isolated, and a failure
# in one does not stop the others.
set -u

SEQ="${1:-floor_EG_2025-12-02_run_1}"
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R"

DS="$R/Data/Hilti/ds/$SEQ"            # our native layout (frames.csv + imu.csv)
EUROC="$R/Data/Hilti/okvis/$SEQ"      # EuRoC layout (cam0/data, imu0/data.csv)
GT="$R/Data/Hilti/groundtruth/$SEQ.txt"
OUT="$R/Data/Hilti/bakeoff/$SEQ"
RUNS="$R/SLAM/experiments"
mkdir -p "$OUT"

UID_GID="$(id -u):$(id -g)"
DK="docker run --rm -u $UID_GID -e HOME=/tmp --cpus=6"
STAMP="$(date +%H:%M:%S)"
echo "=== bake-off on $SEQ  (started $STAMP) ==="

ok()   { echo "  [ok]   $*"; }
skip() { echo "  [SKIP] $*"; }
fail() { echo "  [FAIL] $*"; }

# ---------------------------------------------------------------- OpenVINS ---
run_openvins() {
  local bin="$R/SLAM/build/ws_openvins/devel/.private/ov_msckf/lib/ov_msckf/run_folder"
  [ -x "$bin" ] || { skip "openvins: binary not built"; return; }
  echo "--- openvins"
  $DK -e OV_RANSAC_PX=15 -e OV_MIN_FEAT_PERCENT=1.0 \
    -v "$R/SLAM/build/ws_openvins:/ws" -v "$R/SLAM/third_party/open_vins:/ws/src/open_vins" \
    -v "$R/SLAM/configs/hilti_fast:/config" -v "$R/Data:/data" \
    insv/openvins /ws/devel/lib/ov_msckf/run_folder \
      /config/estimator_config.yaml "/data/Hilti/ds/$SEQ" "/data/Hilti/bakeoff/$SEQ/ov.csv" \
      > "$OUT/openvins.log" 2>&1 \
    && ok "openvins" || fail "openvins (see $OUT/openvins.log)"
}

# ------------------------------------------------------- OKVIS2 / OKVIS2-X ---
# $1 = label, $2 = build dir, $3 = config dir, $4 = "cwd" | "savepath"
# OKVIS2 writes its outputs into the current directory; OKVIS2-X takes an
# explicit save path as argv[3] (argc must be 4 or 5).
run_okvis_family() {
  local label="$1" build="$2" cfgdir="$3" style="$4"
  local bin="$build/okvis_app_synchronous"
  [ -x "$bin" ] || { skip "$label: binary not built ($bin)"; return; }
  echo "--- $label"
  local o="$OUT/$label"; mkdir -p "$o"
  local inner
  if [ "$style" = "savepath" ]; then
    inner="/build/okvis_app_synchronous /config/hilti.yaml /data/Hilti/okvis/$SEQ/ /data/Hilti/bakeoff/$SEQ/$label/"
  else
    inner="cd /data/Hilti/bakeoff/$SEQ/$label && /build/okvis_app_synchronous /config/hilti.yaml /data/Hilti/okvis/$SEQ/"
  fi
  $DK -e QT_QPA_PLATFORM=offscreen \
    -v "$build:/build" -v "$cfgdir:/config" -v "$R/Data:/data" \
    fisheye-slam bash -c "$inner" > "$OUT/$label.log" 2>&1
  # OKVIS2 (non-X) ignores cwd and writes its trajectory/map into the DATASET
  # folder. Move them out, so each estimator's results stay isolated and the
  # next run cannot silently overwrite the previous one's baseline.
  if [ "$style" = "cwd" ]; then
    for f in "$EUROC"/okvis2-slam*.csv "$EUROC"/okvis2-slam*.g2o; do
      [ -e "$f" ] && mv "$f" "$o/"
    done
  fi
  if [ -f "$o/okvis2-slam-final_trajectory.csv" ]; then ok "$label"
  else fail "$label (see $OUT/$label.log)"; fi
}

# --------------------------------------------------------------- ORB-SLAM3 ---
run_orbslam3() {
  local bin="$R/SLAM/third_party/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc"
  [ -x "$bin" ] || { skip "orbslam3: binary not built"; return; }
  echo "--- orbslam3 (mono-inertial, cam0 only)"
  $DK -e QT_QPA_PLATFORM=offscreen \
    -v "$R/SLAM/third_party/ORB_SLAM3:/orb" -v "$R/SLAM/configs/orbslam3_hilti:/config" \
    -v "$R/Data:/data" -v "$OUT:/out" \
    insv/orbslam3 bash -c "cd /out && /orb/Examples/Monocular-Inertial/mono_inertial_euroc \
      /orb/Vocabulary/ORBvoc.txt /config/hilti_mono_inertial.yaml \
      /data/Hilti/okvis/$SEQ /config/timestamps.txt orbslam3" \
    > "$OUT/orbslam3.log" 2>&1 \
    && ok "orbslam3" || fail "orbslam3 (see $OUT/orbslam3.log)"
}

# ------------------------------------------------------------------ Basalt ---
run_basalt() {
  local bin="$R/SLAM/build/basalt_build/basalt_vio"
  [ -x "$bin" ] || { skip "basalt: binary not built"; return; }
  echo "--- basalt"
  $DK -v "$R/SLAM/build/basalt_build:/bb" -v "$R/SLAM/configs/basalt_hilti:/config" \
    -v "$R/Data:/data" -v "$OUT:/out" \
    fisheye-slam /bb/basalt_vio --dataset-path "/data/Hilti/okvis/$SEQ" \
      --cam-calib /config/hilti_calib.json --dataset-type euroc \
      --config-path /config/hilti_config.json --show-gui 0 \
      --result-path /out/basalt_result.json --save-trajectory euroc \
    > "$OUT/basalt.log" 2>&1 \
    && ok "basalt" || fail "basalt (see $OUT/basalt.log)"
}

run_openvins
run_okvis_family okvis2   "$R/SLAM/build/okvis2_build"   "$R/SLAM/configs/okvis_hilti"  cwd
run_okvis_family okvis2x  "$R/SLAM/build/okvis2x_build"  "$R/SLAM/configs/okvis2x_hilti" savepath
# ORB-SLAM3 and Basalt: tomorrow (not built yet)
# run_orbslam3
# run_basalt

# ------------------------------------------------------- run folders + score --
echo
echo "=== building run folders (SLAM/docs/OUTPUT.md contract) ==="
mk() {  # $1 label, then make_run_output args
  local label="$1"; shift
  python SLAM/pipeline/viz/make_run_output.py --dataset "$DS" --gt "$GT" \
    --out "$RUNS/${SEQ}__${label}" "$@" >> "$OUT/viz.log" 2>&1 \
    && ok "run folder: ${SEQ}__${label}" || fail "run folder ${label} (see $OUT/viz.log)"
}
[ -f "$OUT/ov.csv" ] && mk openvins --engine openvins --traj "$OUT/ov.csv" \
    --config "$R/SLAM/configs/hilti_fast"
[ -f "$OUT/okvis2/okvis2-slam-final_trajectory.csv" ] && mk okvis2 --engine okvis \
    --okvis-dir "$OUT/okvis2" --config "$R/SLAM/configs/okvis_hilti"
[ -f "$OUT/okvis2x/okvis2-slam-final_trajectory.csv" ] && mk okvis2x --engine okvis \
    --okvis-dir "$OUT/okvis2x" --config "$R/SLAM/configs/okvis2x_hilti"

echo
echo "=== SUMMARY  ($SEQ) ==="
printf "%-34s %10s %10s %8s\n" "run" "ATE[m]" "cover[%]" "SCORE"
for d in "$RUNS/${SEQ}__"*; do
  [ -f "$d/score.txt" ] || continue
  a=$(grep -oP 'ATE RMSE : \K[0-9.]+' "$d/score.txt")
  c=$(grep -oP 'coverage : \K[0-9.]+' "$d/score.txt")
  s=$(grep -oP 'SCORE    : \K[0-9.]+' "$d/score.txt")
  printf "%-34s %10s %10s %8s\n" "$(basename "$d")" "$a" "$c" "$s"
done
echo
echo "rank-1 reference on this sequence: 97.73"
echo "done at $(date +%H:%M:%S).  rerun <folder>/run.rrd to inspect."
