#!/usr/bin/env bash
# Corrected data-scaling series: subsample TRAIN only (clusters 0,1,2), keep
# valid(3)+test(4) FULL, seeded. 100% point = existing train_run_esm2_100.
# Each run mirrors the ESM2 baseline config (lstmcnncrf, 100 epochs, bs48, lr1e-4).
set -u
cd /home/oskar/work/DeepPeptide
PY=env/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for FRAC in 50 60 70 80 90; do
  OUT="runs/scaling_trainfrac${FRAC}"
  DATA="data/uniprot_2022/scaling/labeled_sequences_trainfrac${FRAC}.csv"
  echo "=== $(date '+%F %T')  START frac${FRAC} -> ${OUT} ==="
  $PY run.py \
    --embedding precomputed \
    --embeddings_dir data/uniprot_2022/embeddings/embeddings_esm2 \
    --data_file "$DATA" \
    --partitioning_file data/uniprot_2022/graphpart_assignments.csv \
    --model lstmcnncrf \
    --label_type multistate_with_propeptides \
    --epochs 100 --batch_size 48 --lr 1e-4 --seed 42 --device 0 \
    --out_dir "$OUT" \
    && echo "=== $(date '+%F %T')  DONE frac${FRAC} ===" \
    || echo "=== $(date '+%F %T')  FAILED frac${FRAC} ==="
done
echo "=== ALL DATA-SCALING RUNS COMPLETE $(date '+%F %T') ==="
