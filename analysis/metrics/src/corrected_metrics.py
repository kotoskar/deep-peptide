#!/usr/bin/env python3
"""Dual-reporting metrics: original (buggy) vs corrected ±3 peptide-finding P/R/F1.

The original DeepPeptide peptide-finding metric
(`manuscript_metrics.get_counts_for_protein`) has a variable-shadowing bug
(inner `for idx, row in pred_df.iterrows()` rebinds `idx`, so the true-side
"matched" flag is written to the row whose label equals the *pred* index — and
when a protein has more predictions than true segments the credit lands on a
phantom row that groupby() then drops, understating recall). This bug is present
in upstream DeepPeptide too, so the published numbers remain comparable to the
paper; we keep them and add a CORRECTED column rather than overwriting.

For every run we run TEST inference, decode peptide/propeptide borders, and
compute P/R/F1 (peptides/propeptides/all, ±3) two ways: the original metric
(as shipped) and a corrected matcher. Output: analysis/corrected_metrics.csv.

Usage: env/bin/python analysis/corrected_metrics.py [--device 0]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())))
import torch
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders, get_counts_for_protein,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)
from analysis.errors.src.error_analysis import run_inference, match_protein

CANON = [r["run"] for r in csv.DictReader(open("analysis/metrics/canonical_metrics.csv"))]


def corrected_counts(true, pred, tol=3):
    g, pm = match_protein(true, pred, tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)


def prf(tp, fn, fp):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_run(run: str, device: str) -> dict:
    run_dir = Path("runs") / run
    preds, names, data, _ = run_inference(run_dir, device)
    # accumulate (tp,fn,fp) per task for both metrics
    orig = {"peptides": [0, 0, 0], "propeptides": [0, 0, 0]}
    corr = {"peptides": [0, 0, 0], "propeptides": [0, 0, 0]}
    states = {"peptides": (PEPTIDE_START_STATE, PEPTIDE_END_STATE),
              "propeptides": (PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)}
    for i, pred in enumerate(preds):
        row = data.loc[names[i]]
        for task, (ss, es) in states.items():
            true = [(int(a), int(b)) for a, b in (row["true_peptides"] if task == "peptides" else row["true_propeptides"])]
            pb = convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)
            otp, ofn, ofp = get_counts_for_protein(true, pb, 3) if (true or pb) else (0, 0, 0)
            ctp, cfn, cfp = corrected_counts(true, pb, 3)
            orig[task][0] += otp; orig[task][1] += ofn; orig[task][2] += ofp
            corr[task][0] += ctp; corr[task][1] += cfn; corr[task][2] += cfp
    out = {"run": run}
    for kind, acc in (("orig", orig), ("corr", corr)):
        allc = [acc["peptides"][k] + acc["propeptides"][k] for k in range(3)]
        for task, c in (("peptides", acc["peptides"]), ("propeptides", acc["propeptides"]), ("all", allc)):
            p, r, f = prf(*c)
            out[f"{kind}_{task}_precision"] = round(p, 6)
            out[f"{kind}_{task}_recall"] = round(r, 6)
            out[f"{kind}_{task}_f1"] = round(f, 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=CANON)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    rows = []
    for run in args.runs:
        if not (Path("runs") / run / "model.pt").exists():
            print(f"[skip] {run}: no model.pt"); continue
        try:
            rows.append(score_run(run, device))
            r = rows[-1]
            print(f"[OK] {run}  f1_all orig={r['orig_all_f1']:.3f} corr={r['corr_all_f1']:.3f} "
                  f"(Δrecall_all={r['corr_all_recall']-r['orig_all_recall']:+.3f})")
        except Exception as e:
            print(f"[FAIL] {run}: {type(e).__name__}: {e}")
    if not rows:
        print("no rows"); return 1
    cols = list(rows[0].keys())
    with open("analysis/metrics/corrected_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nWrote analysis/corrected_metrics.csv ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
