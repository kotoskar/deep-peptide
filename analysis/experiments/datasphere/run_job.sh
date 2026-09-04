#!/bin/bash
# Driver executed INSIDE a DataSphere job. Reassembles the embeddings, runs the
# nested-CV queue, and packs only what has to come home.
#
# Env it reads (set from config.yaml or left at these defaults):
#   OUT_NAME   run name under runs/            default 5cv_baseline_esm2_fp32
#   CONC       cells in flight at once         default 8
#   GPUS       comma-separated GPU ids         default 0
#   PAIRS      subset of "o,k" cells           default all 20
#   EPOCHS     override epochs (smoke runs)    default unset -> the config's 100
set -euo pipefail
# The job captures stdout but not stderr, so a failure in here was invisible:
# the smoke run died silently right after the torch check. Fold stderr in.
exec 2>&1

OUT_NAME="${OUT_NAME:-5cv_baseline_esm2_fp32}"
CONC="${CONC:-8}"
GPUS="${GPUS:-0}"
EMB=data/uniprot_2026/embeddings/emb_esm2

echo "=== $(date -u +%FT%TZ) driver start: out=$OUT_NAME conc=$CONC gpus=$GPUS ==="
nvidia-smi || { echo "no GPU visible -- stopping before spending anything"; exit 1; }
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

# ---- 1. reassemble the embeddings -----------------------------------------
# Shipped as three tars because a single job input file is capped at 5 GiB.
echo "disk before extract:"; df -h . | tail -1
mkdir -p "$EMB"
for t in emb_part0.tar emb_part1.tar emb_part2.tar; do
  [ -f "$t" ] || { echo "missing input $t"; ls -la; exit 1; }
  echo "extracting $t ($(du -h "$t" | cut -f1))"
  tar -xf "$t" -C "$EMB"
  # Free each tar as soon as it is unpacked: keeping all three alongside their
  # contents doubles peak disk for no reason.
  rm -f "$t"
  df -h . | tail -1
done
n=$(ls "$EMB" | wc -l)
want=$(cat analysis/experiments/datasphere/emb_count.txt)
echo "embeddings reassembled: $n files (expected $want)"
# Not the protein count: 461 of the 8,897 proteins in this split share a sequence
# with another, and embeddings are keyed by md5(sequence).
[ "$n" -eq "$want" ] || { echo "expected $want embedding files, got $n"; exit 1; }

# ---- 2. run the queue ------------------------------------------------------
# amp=false is the whole point of this job: the existing 5cv_baseline_esm2 was
# cloned from a base config carrying amp=true, which made it the only bf16 run
# in the grid. Pin it here rather than inheriting it.
EXTRA="amp=false"
[ -n "${EPOCHS:-}" ] && EXTRA="$EXTRA epochs=$EPOCHS"

set +e
MODELS="runs/2026_baseline_esm2/config.json:$OUT_NAME" \
EMB="$EMB" \
SPLIT=data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv \
DATA=data/uniprot_2026/labeled_sequences.csv \
EXTRA_SET="$EXTRA" \
PAIRS="${PAIRS:-}" \
bash analysis/experiments/run_cv_queue.sh "$CONC" "$GPUS"
queue_rc=$?
set -e
echo "queue exited rc=$queue_rc"

# ---- 3. pack only what is needed at home -----------------------------------
# model.pt is included: the corrected-matcher rescoring runs at home from it.
# The periodic checkpoints are not -- they exist only to resume a dying cell.
done_cells=$(find "runs/$OUT_NAME" -name cell_result.json 2>/dev/null | wc -l)
echo "cells finished: $done_cells / 20"
# results.tar must exist even when the queue produced nothing: a declared output
# that is missing fails the upload and buries the reason. Ship the logs too, so a
# failed run still comes back explaining itself.
mkdir -p "runs/$OUT_NAME" logs
find "runs/$OUT_NAME" \( -name 'cell_result.json' -o -name 'config.json' \
     -o -name '*metrics*.json' -o -name 'model.pt' -o -name 'all_metrics.txt' \) \
     -print0 2>/dev/null | tar -cf results.tar --null -T - || tar -cf results.tar --files-from /dev/null
tar -rf results.tar logs 2>/dev/null || true
ls -la results.tar
echo "=== $(date -u +%FT%TZ) driver done ==="
exit "$queue_rc"
