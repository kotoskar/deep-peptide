#!/bin/bash
# Auto-probes how many nested-CV cells can safely train concurrently on this GPU.
# Ramps concurrency 1x -> 2x -> 3x -> 4x -> 6x -> 8x -> 12x -> 16x -> 20x, each stage a
# short time-boxed real-training burst (not full epochs). Stops at the first stage
# with any OOM/crash. Writes the result (one rung back from the observed ceiling,
# as a safety margin) to probe_result.json in the repo root.
#
# Usage: bash probe_concurrency.sh   (run from repo root; takes ~15-30 min)
set -x
cd "$(dirname "$0")/../.." || exit 1   # repo root (this script lives in analysis/experiments/)
mkdir -p probe_logs

BASE=runs/2026_esmc6b_boundary/config.json
EMB=data/uniprot_2022/embeddings/embeddings_esmc6b
SPLIT=data/uniprot_2026/graphpart_assignments_5motif.esmc6bcovered.csv
WINDOW_S=${PROBE_WINDOW_S:-240}   # 4 min: long enough to see steady-state OOMs,
                                  # short enough to never reach the first periodic
                                  # checkpoint (saved every 10 epochs), so re-using
                                  # the same (outer,inner) pairs across stages is safe.
LEVELS=(1 2 3 4 6 8 12 16 20)
PAIRS=(0,1 0,2 0,3 0,4 1,0 1,2 1,3 1,4 2,0 2,1 2,3 2,4 3,0 3,1 3,2 3,4 4,0 4,1 4,2 4,3)

run_probe_cell() {
  local n="$1" o="$2" k="$3"
  python3 analysis/experiments/train_nested_cell.py --base "$BASE" --emb "$EMB" --split "$SPLIT" \
    --out "probe_stage${n}x" --outer "$o" --inner "$k" --n_folds 5 --set epochs=3 \
    > "probe_logs/stage${n}x_o${o}_i${k}.txt" 2>&1
}

SAFE=0
CEILING_HIT=0
for n in "${LEVELS[@]}"; do
  if [ "$n" -gt "${#PAIRS[@]}" ]; then
    echo "Ran out of distinct (outer,inner) pairs before hitting a ceiling -- stopping at max tested level."
    break
  fi
  echo "== probing ${n}x concurrency, ${WINDOW_S}s window =="
  pids=()
  for ((i=0; i<n; i++)); do
    IFS=, read -r o k <<< "${PAIRS[$i]}"
    run_probe_cell "$n" "$o" "$k" &
    pids+=("$!")
  done
  sleep "$WINDOW_S"
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
  wait "${pids[@]}" 2>/dev/null

  fail=0
  for f in probe_logs/stage${n}x_*.txt; do
    if grep -qi "out of memory\|CUDA error\|Traceback" "$f" 2>/dev/null; then
      fail=1
      echo "  FAIL detected in $f"
    fi
  done

  if [ "$fail" = "1" ]; then
    echo "OOM/crash detected at ${n}x -- stopping ramp, ceiling is below ${n}x"
    CEILING_HIT=1
    break
  fi
  SAFE=$n
done

if [ "$SAFE" = "0" ]; then
  echo "Even 1x failed -- something is wrong with the environment (missing deps, bad CUDA setup, etc)."
  echo "Check probe_logs/stage1x_*.txt for the actual error before doing anything else."
  exit 1
fi

# Step back one rung from the observed ceiling as a safety margin (matches how we
# size production runs elsewhere in this project: never run at the exact observed
# limit, back off one notch).
declare -A BACKOFF=( [1]=1 [2]=1 [3]=2 [4]=3 [6]=4 [8]=6 [12]=8 [16]=12 [20]=16 )
if [ "$CEILING_HIT" = "1" ]; then
  RECOMMENDED=${BACKOFF[$SAFE]:-$SAFE}
else
  # never hit a ceiling within tested levels -- use the highest level tested as-is,
  # no need to back off from a limit we never actually observed
  RECOMMENDED=$SAFE
fi

cat > probe_result.json << EOF
{
  "tested_ceiling_no_oom": $SAFE,
  "ceiling_observed": $([ "$CEILING_HIT" = "1" ] && echo true || echo false),
  "recommended_concurrency": $RECOMMENDED,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "=================================================="
echo "Probe done."
echo "  Highest concurrency with NO failures: ${SAFE}x"
echo "  Recommended (with safety margin):     ${RECOMMENDED}x"
echo "  Written to probe_result.json"
echo "=================================================="
