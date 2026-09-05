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
#   MPS        1 to start the MPS daemon       default 0 (see below: worth ~2x)
#   THREADS    cap on per-process CPU threads  default unset (see below: worth ~1.7x)
#
# Which release to train on. The defaults are the 2026 ESM-2 split; the 2022
# release is the same job with four paths and a count changed.
#   EMB_DIR    where the embeddings unpack to  default data/uniprot_2026/embeddings/emb_esm2
#   BASE_CFG   config to clone for every cell  default runs/2026_baseline_esm2/config.json
#   SPLIT_FILE GraphPart assignments csv
#   DATA_FILE  labeled_sequences csv
#   EMB_COUNT  expected unique embedding files default: read from emb_count.txt
set -euo pipefail
# The job captures stdout but not stderr, so a failure in here was invisible:
# the smoke run died silently right after the torch check. Fold stderr in.
exec 2>&1

OUT_NAME="${OUT_NAME:-5cv_baseline_esm2_fp32}"
CONC="${CONC:-8}"
GPUS="${GPUS:-0}"
EMB="${EMB_DIR:-data/uniprot_2026/embeddings/emb_esm2}"
BASE_CFG="${BASE_CFG:-runs/2026_baseline_esm2/config.json}"
SPLIT_FILE="${SPLIT_FILE:-data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv}"
DATA_FILE="${DATA_FILE:-data/uniprot_2026/labeled_sequences.csv}"

echo "=== $(date -u +%FT%TZ) driver start: out=$OUT_NAME conc=$CONC gpus=$GPUS ==="
# Two-stage upload: a job's inputs are capped at 10 GiB, but `job fork` reuses the
# parent's uploaded files and only sends what changed. So the first job carries half
# the embeddings and exits here, and the fork supplies the other half and trains.
if [ "${STAGE_ONLY:-0}" = "1" ]; then
  echo "STAGE_ONLY: inputs uploaded, exiting before the GPU check"
  ls -la
  mkdir -p logs && echo staged > logs/staged.txt
  tar -cf results.tar logs
  echo "=== $(date -u +%FT%TZ) staging done ==="
  exit 0
fi
echo "base=$BASE_CFG emb=$EMB split=$SPLIT_FILE data=$DATA_FILE"
nvidia-smi || { echo "no GPU visible -- stopping before spending anything"; exit 1; }
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

# Ten cells each taking as many intraop threads as there are cores oversubscribes
# the machine: with MPS on, load hit 67.7 on 28 vCPU. Cap the per-process thread
# count when THREADS is set.
if [ -n "${THREADS:-}" ]; then
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
  echo "thread cap: OMP_NUM_THREADS=$THREADS"
fi

# ---- 0. CUDA MPS (optional) ------------------------------------------------
# Without MPS the driver time-slices between processes: kernels from different
# cells never overlap, so ten cells get roughly one cell's throughput plus
# switching overhead. That is exactly what the CONC=10 calibration measured --
# 962 s per epoch against ~180 s for two cells, with 25 GB of VRAM and half the
# cores idle. MPS lets the kernels run concurrently, which for a 224k-parameter
# model should be a multiple rather than a few percent.
if [ "${MPS:-0}" = "1" ]; then
  if command -v nvidia-cuda-mps-control >/dev/null; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if nvidia-cuda-mps-control -d; then
      echo "MPS daemon started"
    else
      echo "MPS daemon FAILED to start -- continuing without it"
    fi
  else
    echo "MPS: nvidia-cuda-mps-control not present -- continuing without it"
  fi
fi

# ---- 0. resource sampler ---------------------------------------------------
# The console shows peaks only while the job is alive, and the first smoke run
# came home with no usage numbers at all. Sample them ourselves, every 10 s, so
# sizing the next run does not depend on someone watching a dashboard.
mkdir -p logs
{
  while true; do
    ts=$(date +%s)
    gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
          --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
    read -r _ mem_total _ < <(grep MemTotal /proc/meminfo)
    read -r _ mem_avail _ < <(grep MemAvailable /proc/meminfo)
    load=$(cut -d' ' -f1 /proc/loadavg)
    echo -e "${ts}\t${gpu}\tram_used_mb=$(( (mem_total - mem_avail) / 1024 ))\tload=${load}"
    sleep 10
  done
} >> logs/usage.tsv 2>&1 &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# ---- 1. reassemble the embeddings -----------------------------------------
# Shipped as three tars because a single job input file is capped at 5 GiB.
echo "disk before extract:"; df -h . | tail -1
mkdir -p "$EMB"
# Glob rather than a fixed list: a second release ships its own tars under its
# own prefix, and how many parts it took is a property of that release's size.
mapfile -t tars < <(find . -name 'emb*part*.tar' | sort)
[ "${#tars[@]}" -gt 0 ] || { echo "no emb*part*.tar inputs found"; ls -laR | head -50; exit 1; }
echo "found ${#tars[@]} embedding tars: ${tars[*]}"
for t in "${tars[@]}"; do
  echo "extracting $t ($(du -h "$t" | cut -f1))"
  tar -xf "$t" -C "$EMB"
  # Free each tar as soon as it is unpacked: keeping all three alongside their
  # contents doubles peak disk for no reason.
  rm -f "$t"
  df -h . | tail -1
done
n=$(ls "$EMB" | wc -l)
want="${EMB_COUNT:-$(cat analysis/experiments/datasphere/emb_count.txt)}"
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
MODELS="$BASE_CFG:$OUT_NAME" \
EMB="$EMB" \
SPLIT="$SPLIT_FILE" \
DATA="$DATA_FILE" \
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
kill $SAMPLER_PID 2>/dev/null || true
[ "${MPS:-0}" = "1" ] && { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
mkdir -p "runs/$OUT_NAME" logs
echo "--- peak usage over the run ---"
awk -F'\t' '
  { for (i = 2; i < NF - 1; i++) { n = split($i, g, /, */); if (n >= 3) {
        if (g[2] + 0 > util[g[1]]) util[g[1]] = g[2] + 0
        if (g[3] + 0 > vram[g[1]]) vram[g[1]] = g[3] + 0 } }
    sub(/ram_used_mb=/, "", $(NF-1)); if ($(NF-1) + 0 > ram) ram = $(NF-1) + 0
    sub(/load=/, "", $NF);           if ($NF + 0 > ld)      ld  = $NF + 0 }
  END { for (i in vram) printf "gpu%s peak util %d%% peak vram %d MiB\n", i, util[i], vram[i]
        printf "peak ram %d MiB, peak 1-min load %.1f, %d samples\n", ram, ld, NR }
' logs/usage.tsv 2>/dev/null || true

# One tar built from one file list: `tar -r` is unavailable in a busybox tar and
# that is why logs/ silently never reached home from the smoke run.
{ find "runs/$OUT_NAME" \( -name 'cell_result.json' -o -name 'config.json' \
       -o -name '*metrics*.json' -o -name 'model.pt' -o -name 'all_metrics.txt' \) 2>/dev/null
  find logs -type f 2>/dev/null
} > /tmp/results_files.txt
tar -cf results.tar -T /tmp/results_files.txt || tar -cf results.tar --files-from /dev/null
ls -la results.tar; tar -tf results.tar | wc -l
echo "=== $(date -u +%FT%TZ) driver done ==="
exit "$queue_rc"
