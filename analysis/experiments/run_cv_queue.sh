#!/bin/bash
# Generic 5x4 nested-CV queue: trains every (outer, inner) cell of one or more
# models to real completion (100 epochs, best-valid checkpoint), CONC cells at a
# time, spread round-robin over the GPUs in GPUS. Batches wait for full
# completion before the next batch starts. Reboot-safe: finished cells
# (cell_result.json present) are skipped, unfinished ones resume from their last
# checkpoint (saved every 10 epochs).
#
# Usage (from the archive/repo root, venv activated):
#   bash analysis/experiments/run_cv_queue.sh [CONC] [GPUS]
#   CONC=4 GPUS=0,1 bash analysis/experiments/run_cv_queue.sh
#
#   CONC   cells running at the same time (all GPUs together). Default: the
#          recommended_concurrency from probe_result.json, else 1.
#   GPUS   comma-separated GPU ids to spread the cells over. Default: 0.
#          A cell gets CUDA_VISIBLE_DEVICES=<one id>, so the config's "device": 0
#          always means "the GPU this cell was given".
#
# Everything else is an env var with a baseline-on-2022 default:
#   MODELS     space-separated "<base config>:<output name>" pairs
#   EMB        embeddings dir (md5(sequence).pt files)
#   SPLIT      GraphPart assignment csv with the 5 folds (column `cluster`)
#   DATA       labeled_sequences.csv of the same release
#   EXTRA_SET  extra "k=v" config overrides passed to every cell, e.g.
#              EXTRA_SET="amp=false" on a GPU without bf16 (V100 and older)
#   PAIRS      subset of "o,k" cells to run (default: all 20), e.g. PAIRS="0,1 0,2"
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1   # archive/repo root
mkdir -p logs

CONC="${1:-${CONC:-}}"
GPUS="${2:-${GPUS:-0}}"
if [ -z "$CONC" ]; then
  if [ -f probe_result.json ]; then
    CONC=$(python3 -c "import json; print(json.load(open('probe_result.json'))['recommended_concurrency'])")
  else
    CONC=1
  fi
fi
if ! [ "$CONC" -ge 1 ] 2>/dev/null; then
  echo "CONC must be a positive integer, got '$CONC'"; exit 1
fi

MODELS="${MODELS:-runs/baseline_esm2/config.json:5cv_baseline_esm2_2022}"
EMB="${EMB:-data/uniprot_2022/embeddings/embeddings_esm2}"
SPLIT="${SPLIT:-data/uniprot_2022/graphpart_assignments.csv}"
DATA="${DATA:-data/uniprot_2022/labeled_sequences.csv}"
EXTRA_SET="${EXTRA_SET:-}"
export NO_AIM=1

IFS=, read -r -a GPU_LIST <<< "$GPUS"
echo "concurrency=$CONC gpus=${GPU_LIST[*]} models=$MODELS"
echo "emb=$EMB split=$SPLIT data=$DATA extra_set='$EXTRA_SET'"
for f in "$EMB" "$SPLIT" "$DATA"; do
  [ -e "$f" ] || { echo "missing input: $f"; exit 1; }
done

run_cell() {
  local base="$1" out="$2" o="$3" k="$4" gpu="$5"
  if [ -f "runs/${out}/outer${o}_inner${k}/cell_result.json" ]; then
    echo "[skip done] ${out} outer${o}_inner${k}"
    return
  fi
  local set_args=()
  [ -n "$EXTRA_SET" ] && set_args=(--set $EXTRA_SET)
  CUDA_VISIBLE_DEVICES="$gpu" python3 analysis/experiments/train_nested_cell.py \
    --base "$base" --emb "$EMB" --split "$SPLIT" --data "$DATA" \
    --out "$out" --outer "$o" --inner "$k" --n_folds 5 "${set_args[@]}" \
    > "logs/${out}_log_o${o}_i${k}.txt" 2>&1
  if [ ! -f "runs/${out}/outer${o}_inner${k}/cell_result.json" ]; then
    echo "FAIL ${out} outer${o}_inner${k} (gpu $gpu), see logs/${out}_log_o${o}_i${k}.txt" | tee -a logs/failures.txt
  fi
}

# PAIRS env limits the run to a subset of (outer,inner) cells, e.g. PAIRS="0,1 0,2"
read -r -a PAIRS <<< "${PAIRS:-0,1 0,2 0,3 0,4 1,0 1,2 1,3 1,4 2,0 2,1 2,3 2,4 3,0 3,1 3,2 3,4 4,0 4,1 4,2 4,3}"

# model-major inside each (outer, inner) pair, so an interruption still leaves
# every model covered across a spread of folds
JOBS=()
for pair in "${PAIRS[@]}"; do
  for m in $MODELS; do
    JOBS+=("$m:$pair")
  done
done

n=${#JOBS[@]}
ng=${#GPU_LIST[@]}
for ((start=0; start<n; start+=CONC)); do
  pids=()
  echo "BATCH start=$start conc=$CONC $(date +%s)" | tee -a logs/timing.txt
  for ((i=start; i<start+CONC && i<n; i++)); do
    IFS=: read -r base out opair <<< "${JOBS[$i]}"
    IFS=, read -r o k <<< "$opair"
    gpu="${GPU_LIST[$(( (i - start) % ng ))]}"
    run_cell "$base" "$out" "$o" "$k" "$gpu" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  echo "BATCH end=$start $(date +%s)" | tee -a logs/timing.txt
done

echo "ALL DONE" | tee -a logs/timing.txt
if [ -s logs/failures.txt ]; then
  echo "$(wc -l < logs/failures.txt) cell(s) failed, see logs/failures.txt" | tee -a logs/timing.txt
fi
for m in $MODELS; do
  IFS=: read -r _ out <<< "$m"
  python3 analysis/experiments/aggregate_cv.py "$out" || true
done
