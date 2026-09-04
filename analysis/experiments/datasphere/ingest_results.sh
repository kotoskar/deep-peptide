#!/bin/bash
# Turn a finished DataSphere results.tar into every number and figure that
# depends on it. Run from the repo root, on a machine with the GPU: the
# tolerance sweep re-runs test-partition inference from each cell's model.pt.
#
#   bash analysis/experiments/datasphere/ingest_results.sh [results.tar] [run-name]
#
# Default run name is 5cv_baseline_esm2_fp32. Nothing here overwrites the bf16
# baseline: the fp32 run lives beside it under its own name, so the two can be
# compared before anything in the paper is repointed.
set -euo pipefail
TAR="${1:-results.tar}"
NAME="${2:-5cv_baseline_esm2_fp32}"
PY=env/bin/python

[ -f "$TAR" ] || { echo "no $TAR here"; exit 1; }
echo "=== 1. unpack ==="
tar -xf "$TAR"
n=$(find "runs/$NAME" -name cell_result.json 2>/dev/null | wc -l)
echo "cells present: $n / 20"
[ "$n" -eq 20 ] || echo "WARNING: incomplete grid -- everything below will be an average over $n cells"

echo "=== 2. sanity: every cell must be fp32 and seed 42 ==="
$PY - "$NAME" <<'PYEOF'
import json, glob, sys, collections
name = sys.argv[1]
c = collections.Counter()
for f in glob.glob(f"runs/{name}/outer*_inner*/config.json"):
    d = json.load(open(f))
    c[(d.get("amp"), d.get("seed"), d.get("epochs"))] += 1
print("  (amp, seed, epochs) ->", dict(c))
bad = [k for k in c if k[0] is not False]
if bad:
    raise SystemExit(f"  ABORT: cells with amp != false: {bad}")
print("  ok: every cell trained in fp32")
PYEOF

echo "=== 3. corrected-matcher rescoring ==="
$PY analysis/experiments/rescore_nested_cv_corrected.py --names "$NAME"

echo "=== 4. tolerance sweep (inference from model.pt, dumps segments) ==="
$PY analysis/metrics/src/tolerance_sweep_cv.py --models "$NAME"

echo "=== 5. everything that reads the segment dumps ==="
$PY analysis/metrics/src/segment_quality_cv.py \
    --models 5cv_baseline_esm2 "$NAME" 5cv_esm2_boundary 5cv_esm2_adapter_only 5cv_esm2_full 5cv_esmc6b_plain \
    --baseline "$NAME" --out analysis/metrics/segment_quality_cv_fp32.json
$PY analysis/metrics/src/bylength_cv.py \
    --models "$NAME" 5cv_esm2_boundary 5cv_esm2_adapter_only 5cv_esm2_full 5cv_esmc6b_plain \
    --out analysis/metrics/bylength_cv_fp32.json
$PY analysis/metrics/src/boundary_error_cv.py --baseline "$NAME" \
    --variants 5cv_esm2_boundary 5cv_esm2_adapter_only 5cv_esm2_full 5cv_esmc6b_plain \
    --out analysis/metrics/fp32

echo "=== 6. what changed, bf16 baseline vs fp32 baseline ==="
$PY - "$NAME" <<'PYEOF'
import json, sys
new = sys.argv[1]
old = "5cv_baseline_esm2"
a = json.load(open(f"runs/{old}/nested_cv_tolerance.json"))
b = json.load(open(f"runs/{new}/nested_cv_tolerance.json"))
print(f"  {'':10s} {'bf16':>18s} {'fp32':>18s}   delta")
for t in (3, 2, 1, 0):
    k = f"cv_tol{t}_all_f1"
    print(f"  tol{t:<7d} {a[k+'_mean']:.4f}+-{a[k+'_std']:.4f}   {b[k+'_mean']:.4f}+-{b[k+'_std']:.4f}   {b[k+'_mean']-a[k+'_mean']:+.4f}")
print("\n  effects against the NEW fp32 baseline at +-3:")
base = b["cv_tol3_all_f1_mean"]
for r, lab in (("5cv_esm2_boundary", "+ head"), ("5cv_esm2_adapter_only", "+ adapter"),
               ("5cv_esm2_full", "+ both"), ("5cv_esmc6b_plain", "ESM-C 6B")):
    d = json.load(open(f"runs/{r}/nested_cv_tolerance.json"))
    po_b, po_n = b["per_outer_tol3_all_f1"], d["per_outer_tol3_all_f1"]
    wins = sum(po_n[f] > po_b[f] for f in po_b)
    print(f"    {lab:10s} {d['cv_tol3_all_f1_mean']-base:+.4f}   beats the base on {wins}/5 outer folds")
PYEOF

echo "=== 7. figures ==="
$PY analysis/metrics/src/generators/fig_tolerance_cv.py --outdir texs/ai4dd/figures
$PY analysis/metrics/src/generators/paper/fig_bylength_cv.py
echo
echo "Figure 1 (fig_hook_cv.py) still points at 5cv_baseline_esm2 by name; repoint"
echo "MODELS/EX_BASE there once the fp32 run is the one the paper reports."
echo "=== done ==="
