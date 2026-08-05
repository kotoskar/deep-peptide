#!/bin/bash
# Runs the remaining 3 esmc6b nested-CV models (boundary, adapter_only, full;
# 20 cells each = 60 total) to real completion, at the concurrency recorded by
# probe_concurrency.sh in probe_result.json. Batches wait for full completion
# (real 100-epoch training, not time-boxed) before starting the next batch.
#
# Usage: bash run_esmc_queue.sh   (run from repo root, AFTER probe_concurrency.sh)
set -x
cd "$(dirname "$0")/../.." || exit 1   # repo root
mkdir -p logs

if [ ! -f probe_result.json ]; then
  echo "probe_result.json not found -- run probe_concurrency.sh first."
  exit 1
fi
CONC=$(python3 -c "import json; print(json.load(open('probe_result.json'))['recommended_concurrency'])")
if [ -z "$CONC" ] || [ "$CONC" -lt 1 ]; then
  echo "Could not read a valid concurrency from probe_result.json"
  exit 1
fi
echo "Using concurrency=$CONC (from probe_result.json)"

EMB=data/uniprot_2022/embeddings/embeddings_esmc6b
SPLIT=data/uniprot_2026/graphpart_assignments_5motif.esmc6bcovered.csv

run_cell() {
  local base="$1" out="$2" o="$3" k="$4"
  # Skip work that's already done (e.g. this is a re-run after an earlier partial run).
  if [ -f "runs/${out}/outer${o}_inner${k}/cell_result.json" ]; then
    echo "[skip done] ${out} outer${o}_inner${k}"
    return
  fi
  python3 analysis/experiments/train_nested_cell.py --base "$base" --emb "$EMB" --split "$SPLIT" \
    --out "$out" --outer "$o" --inner "$k" --n_folds 5 \
    > "logs/${out}_log_o${o}_i${k}.txt" 2>&1
  if [ ! -f "runs/${out}/outer${o}_inner${k}/cell_result.json" ]; then
    echo "FAIL ${out} outer${o}_inner${k}" | tee -a logs/failures.txt
  fi
}

# 3 models x 20 cells = 60 jobs, interleaved model-major within each (outer,inner)
# pair so an interruption still covers all 3 models across a spread of folds.
MODELS=("runs/2026_esmc6b_boundary/config.json:5cv_esmc6b_boundary" "runs/2026_esmc6b_adapter_only/config.json:5cv_esmc6b_adapter_only" "runs/2026_esmc6b_full/config.json:5cv_esmc6b_full")
PAIRS=(0,1 0,2 0,3 0,4 1,0 1,2 1,3 1,4 2,0 2,1 2,3 2,4 3,0 3,1 3,2 3,4 4,0 4,1 4,2 4,3)

JOBS=()
for pair in "${PAIRS[@]}"; do
  for m in "${MODELS[@]}"; do
    JOBS+=("$m:$pair")
  done
done

n=${#JOBS[@]}
for ((start=0; start<n; start+=CONC)); do
  pids=()
  echo "BATCH start=$start conc=$CONC $(date +%s)" | tee -a logs/timing.txt
  for ((i=start; i<start+CONC && i<n; i++)); do
    IFS=: read -r base out opair <<< "${JOBS[$i]}"
    IFS=, read -r o k <<< "$opair"
    run_cell "$base" "$out" "$o" "$k" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  echo "BATCH end=$start $(date +%s)" | tee -a logs/timing.txt
done

echo "ALL DONE" | tee -a logs/timing.txt
if [ -s logs/failures.txt ]; then
  echo "$(wc -l < logs/failures.txt) cell(s) failed, see logs/failures.txt" | tee -a logs/timing.txt
fi
