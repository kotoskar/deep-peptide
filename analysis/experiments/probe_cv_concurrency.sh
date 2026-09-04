#!/bin/bash
# Probes how many nested-CV cells can train concurrently on ONE GPU without OOM.
# Ramps 1x -> 2x -> 3x -> 4x -> 6x -> 8x -> 12x -> 16x -> 20x, each stage a short
# time-boxed burst of real training, stops at the first OOM/crash, and writes
# probe_result.json (one rung back from the ceiling as a margin). run_cv_queue.sh
# reads that file when CONC is not given explicitly.
#
# Usage (from the archive/repo root, venv activated): bash analysis/experiments/probe_cv_concurrency.sh
# Env: BASE, EMB, SPLIT, DATA, EXTRA_SET as in run_cv_queue.sh; GPU (default 0);
#      PROBE_WINDOW_S (default 240). Takes ~15-30 min.
#
# Result is per GPU. With N identical GPUs, run the queue with CONC = N x
# recommended and GPUS=<all ids>.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p probe_logs

BASE="${BASE:-runs/baseline_esm2/config.json}"
EMB="${EMB:-data/uniprot_2022/embeddings/embeddings_esm2}"
SPLIT="${SPLIT:-data/uniprot_2022/graphpart_assignments.csv}"
DATA="${DATA:-data/uniprot_2022/labeled_sequences.csv}"
EXTRA_SET="${EXTRA_SET:-}"
GPU="${GPU:-0}"
WINDOW_S=${PROBE_WINDOW_S:-240}   # never reaches the first periodic checkpoint
                                  # (every 10 epochs), so probe dirs stay disposable
export NO_AIM=1
LEVELS=(1 2 3 4 6 8 12 16 20)
PAIRS=(0,1 0,2 0,3 0,4 1,0 1,2 1,3 1,4 2,0 2,1 2,3 2,4 3,0 3,1 3,2 3,4 4,0 4,1 4,2 4,3)

run_probe_cell() {
  local n="$1" o="$2" k="$3"
  local set_args=(--set epochs=3)
  [ -n "$EXTRA_SET" ] && set_args=(--set epochs=3 $EXTRA_SET)
  CUDA_VISIBLE_DEVICES="$GPU" python3 analysis/experiments/train_nested_cell.py \
    --base "$BASE" --emb "$EMB" --split "$SPLIT" --data "$DATA" \
    --out "probe_stage${n}x" --outer "$o" --inner "$k" --n_folds 5 "${set_args[@]}" \
    > "probe_logs/stage${n}x_o${o}_i${k}.txt" 2>&1
}

SAFE=0
CEILING_HIT=0
for n in "${LEVELS[@]}"; do
  echo "== probing ${n}x concurrency on GPU $GPU, ${WINDOW_S}s window =="
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
    echo "OOM/crash at ${n}x -- ceiling is below ${n}x"
    CEILING_HIT=1
    break
  fi
  SAFE=$n
done
rm -rf runs/probe_stage*x

if [ "$SAFE" = "0" ]; then
  echo "Even 1x failed -- environment problem (deps, CUDA, bf16 on an old GPU?)."
  echo "Read probe_logs/stage1x_*.txt. If it mentions bf16/bfloat16, retry with EXTRA_SET=amp=false."
  exit 1
fi

declare -A BACKOFF=( [1]=1 [2]=1 [3]=2 [4]=3 [6]=4 [8]=6 [12]=8 [16]=12 [20]=16 )
if [ "$CEILING_HIT" = "1" ]; then
  RECOMMENDED=${BACKOFF[$SAFE]:-$SAFE}
else
  RECOMMENDED=$SAFE
fi

cat > probe_result.json << EOF
{
  "tested_ceiling_no_oom": $SAFE,
  "ceiling_observed": $([ "$CEILING_HIT" = "1" ] && echo true || echo false),
  "recommended_concurrency": $RECOMMENDED,
  "gpu": "$GPU",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "=================================================="
echo "  Highest concurrency with NO failures: ${SAFE}x"
echo "  Recommended (with safety margin):     ${RECOMMENDED}x  (per GPU)"
echo "  Written to probe_result.json"
echo "=================================================="
