"""Adversarial check of the 'ESM-C adapter number is an upper estimate because fold 2 is
missing' claim.  All figures at +-3 with the corrected matcher (tol3_all_f1 in
tolerance_metrics.json), aggregated micro-in-cell -> mean over inner -> mean over outer.
Run from repo root with ./env/bin/python."""
import json, glob, re, statistics as st

def load(run):
    d = {}
    for p in glob.glob(f"runs/{run}/outer*_inner*/tolerance_metrics.json"):
        o, i = map(int, re.search(r"outer(\d+)_inner(\d+)", p).groups())
        d[(o, i)] = json.load(open(p))["tol3_all_f1"]
    return d

A = load("5cv_esmc6b_adapter_only")
P = load("5cv_esmc6b_plain")
NINE = [(0,1),(0,2),(0,3),(0,4),(1,0),(1,2),(1,3),(3,4),(4,0)]  # cells with cell_result_corrected.json

def outer_means(d, cells):
    per = {}
    for c in cells:
        per.setdefault(c[0], []).append(d[c])
    return {o: sum(v)/len(v) for o, v in per.items()}

def paired(cells):
    a, p = outer_means(A, cells), outer_means(P, cells)
    dl = [a[o] - p[o] for o in sorted(a)]
    m, s = sum(dl)/len(dl), st.stdev(dl)
    return dl, m, s, m/(s/len(dl)**0.5)

for label, cells in [("9-cell set (as claimed)", NINE), ("all cells on disk", sorted(A))]:
    dl, m, s, t = paired(cells)
    a = outer_means(A, cells)
    print(f"{label}: n_cells={len(cells)} folds={sorted(a)}")
    print(f"  adapter abs = {sum(a.values())/len(a):.4f}")
    print(f"  paired effect vs plain = +{m:.4f} +/- {s:.4f}  {sum(x>0 for x in dl)}/{len(dl)} positive  t={t:.2f}")

# RESULT (run completed to 20/20 during the 2026-09-06 check):
#   9-cell provisional : abs 0.6067, paired effect +0.0162 +/- 0.0163, 4/4, t=1.98
#   complete 20/20     : abs 0.6016, paired effect +0.0139 +/- 0.0141, 5/5, t=2.20
#   fold-2 removal alone lifts plain +0.0042 of the +0.0028 net 9-cell offset.
#   NOTE: runs/5cv_esmc6b_adapter_only/nested_cv_tolerance.json and
#   analysis/metrics/ALL_NUMBERS.txt still carry the stale n_cells=9 aggregate.
