#!/bin/bash
# Ingest whatever nested-CV runs have landed, in the right order, and report what
# changed. Safe to re-run: every stage skips work that is already done.
#
#   bash analysis/experiments/datasphere/ingest_any.sh 5cv_esmc6b_boundary 5cv_esmc6b_full ...
#   bash analysis/experiments/datasphere/ingest_any.sh            # auto-detect new runs
#
# "New" means a runs/5cv_* directory with cell_result.json files but no
# nested_cv_tolerance.json, i.e. trained but never scored here.
#
# Needs the local GPU: the tolerance sweep re-runs test-partition inference from
# each cell's model.pt. Everything after that is CPU.
set -uo pipefail
PY=env/bin/python

if [ "$#" -gt 0 ]; then
  RUNS=("$@")
else
  mapfile -t RUNS < <(for d in runs/5cv_*; do
      [ -d "$d" ] || continue
      n=$(find "$d" -name cell_result.json 2>/dev/null | wc -l)
      [ "$n" -gt 0 ] && [ ! -f "$d/nested_cv_tolerance.json" ] && basename "$d"
    done)
fi
[ "${#RUNS[@]}" -gt 0 ] || { echo "nothing new to ingest"; exit 0; }

echo "=== runs to ingest: ${RUNS[*]} ==="
for r in "${RUNS[@]}"; do
  n=$(find "runs/$r" -name cell_result.json 2>/dev/null | wc -l)
  amp=$($PY -c "
import json,glob,collections
c=collections.Counter(json.load(open(f)).get('amp') for f in glob.glob('runs/$r/outer*_inner*/config.json'))
print(dict(c))" 2>/dev/null)
  echo "  $r: $n/20 cells, amp=$amp"
done
echo

for r in "${RUNS[@]}"; do
  echo "=== $r: corrected-matcher rescoring ==="
  $PY analysis/experiments/rescore_nested_cv_corrected.py --names "$r" || echo "  rescore FAILED for $r, continuing"
  echo "=== $r: tolerance sweep (GPU inference from model.pt) ==="
  $PY analysis/metrics/src/tolerance_sweep_cv.py --models "$r" || echo "  sweep FAILED for $r, continuing"
done

# Everything below reads the segment dumps, so it runs once over all runs that have them.
mapfile -t SCORED < <(for d in runs/5cv_*; do
    [ -f "$d/nested_cv_tolerance.json" ] && basename "$d"; done)
echo
echo "=== scored runs now available: ${SCORED[*]} ==="

echo "=== +-3 headline, every scored run ==="
$PY - "${SCORED[@]}" <<'PYEOF'
import json, sys
rows=[]
for r in sys.argv[1:]:
    d=json.load(open(f"runs/{r}/nested_cv_tolerance.json"))
    rows.append((r, d["cv_tol3_all_f1_mean"], d["cv_tol3_all_f1_std"], d["cv_tol0_all_f1_mean"],
                 d.get("per_outer_tol3_all_f1", {})))
rows.sort(key=lambda x: -x[1])
print(f"  {'run':30s} {'F1@+-3':>16s} {'F1@exact':>9s}")
for r,m,s,e,_ in rows: print(f"  {r:30s} {m:.4f}+-{s:.4f}   {e:.4f}")
# every pairwise per-fold win count against whichever baseline is present
base = next((r for r,_,_,_,_ in rows if r in ("5cv_baseline_esm2_fp32","5cv_baseline_esm2")), None)
if base:
    pb = next(p for r,_,_,_,p in rows if r==base)
    print(f"\n  per-outer-fold wins against {base}:")
    for r,m,s,e,p in rows:
        if r==base or not p: continue
        w=sum(p[f]>pb[f] for f in pb)
        print(f"    {r:30s} {m-next(x[1] for x in rows if x[0]==base):+.4f}   {w}/5 folds")
PYEOF

echo
echo "=== boundary metrics over every scored run ==="
$PY analysis/metrics/src/segment_quality_cv.py --models "${SCORED[@]}" || true
$PY analysis/metrics/src/bylength_cv.py --models "${SCORED[@]}" || true

echo "=== figures ==="
$PY analysis/metrics/src/generators/fig_tolerance_cv.py --outdir texs/ai4dd/figures || true
$PY analysis/metrics/src/generators/paper/fig_bylength_cv.py || true
$PY analysis/metrics/src/generators/fig_hook_cv.py || true
echo
echo "Both figure generators already list the ESM-C 6B variants and skip missing dirs,"
echo "so they pick the new runs up on their own. branch_b_esmc.tex holds the table rows"
echo "and the per-scenario sentences for the ESM-C grid."
echo "=== done ==="
