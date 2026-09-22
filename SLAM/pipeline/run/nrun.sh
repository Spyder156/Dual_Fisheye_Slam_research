#!/usr/bin/env bash
# N-run variance harness. The estimator is nondeterministic (thread
# interleaving decides whether a run survives the dark window), so a single
# run is a lottery ticket, not a measurement. Every change from here on is
# judged on N runs: median score, min/max, and survival.
#
#   nrun.sh <config.yaml> <name> <N> [secs|full] [seq]
#
#   config : path relative to repo root (e.g. SLAM/configs/orbslam3_hilti/monorig_carto.yaml)
#   name   : experiment name -> SLAM/experiments/<name>/r01..rNN + summary
#   N      : number of sequential runs
#   secs   : 20/30/40 = short window (plumbing checks), "full" = whole sequence
#   seq    : dataset sequence (default floor_EG_2025-12-02_run_2)
#
# Runs are SEQUENTIAL on purpose: parallel runs would contend for cores and
# change the very thread timing we are trying to measure.
set -uo pipefail
W="$(cd "$(dirname "$0")/../../.." && pwd)"
CFG="$1"; NAME="$2"; N="$3"; SPAN="${4:-full}"; SEQ="${5:-floor_EG_2025-12-02_run_2}"
ROOT="SLAM/experiments/$NAME"
mkdir -p "$W/$ROOT"

if [ "$SPAN" = "full" ]; then
  TS="/ws/SLAM/configs/orbslam3_hilti/timestamps_${SEQ}.txt"
else
  # short windows only exist for run_2 (timestamps_run2_first{20,30,40}s.txt)
  TS="/ws/SLAM/configs/orbslam3_hilti/timestamps_run2_first${SPAN}s.txt"
fi
GT="$W/Data/Hilti/groundtruth/${SEQ}.txt"

# Data/ + experiments are absolute symlinks onto the 4TB drive; mount the
# drive at the same path inside the container (see short_test.sh).
DRIVE="$(readlink -f "$W/Data")"; DRIVE="${DRIVE%/INSV_STITCHING/Data}"

SUM="$W/$ROOT/summary.csv"
echo "run,score,ate_m,coverage_pct,track_fails,map_resets,wall_s" > "$SUM"

for i in $(seq 1 "$N"); do
  R="$ROOT/r$(printf '%02d' "$i")"
  mkdir -p "$W/$R"
  echo "=== run $i/$N -> $R"
  t0=$SECONDS
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -w "/ws/$R" -v "$W:/ws" -v "$DRIVE:$DRIVE" \
    -v "$W/SLAM/configs/orbslam3_hilti:/config" \
    insv/orbslam3 bash -c "
    export LD_LIBRARY_PATH=/ws/SLAM/third_party/ORB_SLAM3/lib:/ws/SLAM/third_party/ORB_SLAM3/Thirdparty/DBoW2/lib:/ws/SLAM/third_party/ORB_SLAM3/Thirdparty/g2o/lib:\$LD_LIBRARY_PATH
    stdbuf -oL -eL /ws/SLAM/third_party/ORB_SLAM3/Examples/Monocular-Inertial/mono_rig_euroc \
      /ws/SLAM/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
      /ws/$CFG /ws/Data/Hilti/orb/$SEQ $TS orb > log.txt 2>&1"
  wall=$((SECONDS - t0))

  LOG="$W/$R/log.txt"
  # grep -c prints the 0 itself on no-match (exit 1); only a missing file
  # needs the fallback, so guard on the file, not the exit code
  fails=0; resets=0
  if [ -f "$LOG" ]; then
    fails=$(grep -c "Fail to track local map" "$LOG" || true)
    resets=$(grep -c "LM: Active map reset recieved" "$LOG" || true)
  fi

  if [ -s "$W/$R/f_orb.txt" ]; then
    line=$(python "$W/SLAM/pipeline/run/hilti_score.py" "$W/$R/f_orb.txt" "$GT" \
           | tee "$W/$R/score.txt" | awk '/^CSV /{print $2}')
  else
    line=""
    echo "  [FAIL] no f_orb.txt (crash? see $R/log.txt)"
  fi
  [ -n "$line" ] || line="nan,nan,nan"
  echo "$i,$line,$fails,$resets,$wall" >> "$SUM"
  echo "  score,ate,cov = $line   fails=$fails resets=$resets  (${wall}s)"
done

python - "$SUM" <<'EOF'
import sys, numpy as np
rows = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
rows = np.atleast_1d(rows)
s = rows["score"]; ok = ~np.isnan(s)
print("\n===== summary =====")
print(f"runs      : {len(s)}   finished: {ok.sum()}")
if ok.sum():
    print(f"score     : median {np.median(s[ok]):.2f}   min {s[ok].min():.2f}   max {s[ok].max():.2f}")
    print(f"ATE [m]   : median {np.median(rows['ate_m'][ok]):.3f}   worst {rows['ate_m'][ok].max():.3f}")
    print(f"coverage  : median {np.median(rows['coverage_pct'][ok]):.1f} %")
    print(f"fails     : {rows['track_fails'].astype(int).tolist()}")
    print(f"resets    : {rows['map_resets'].astype(int).tolist()}")
EOF
echo "-> $ROOT/summary.csv"
