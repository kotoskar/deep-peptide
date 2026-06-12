#!/usr/bin/env python3
"""Threshold (cleavage-tolerance) sweep: F1 vs acceptance window 0..5 per model.

Hypothesis (roadmap #2): a boundary/bond head sharpens cleavage sites, so it may
be MORE robust at TIGHTER acceptance windows (0-1) than baselines — an advantage
the fixed ±3 metric hides. We sweep the window and plot F1 vs window per model.

Matcher: the CORRECTED ±3-style matcher (`match_protein` from error_analysis.py)
applied at each tolerance 0..5 — deliberately NOT `compute_all_metrics`, which
rides the shadow-bug `get_counts_for_protein`. A sweep is literally a study of
matching-vs-window, so the matcher must be the unbiased one, and all models must
use the SAME matcher for a fair cross-model comparison. (The buggy curve can be
added later as an overlay to prove the conclusion isn't a matcher artifact.)

Inference is fp32 (run_dataloader default use_amp=False), matching the infer.py
eval that produced big_metrics_table. RECONCILE: F1 at tol=3 must reproduce each
model's "new" F1 in analysis/metrics/big_metrics_table.csv (same corrected matcher).

Outputs:
  analysis/errors/error_stats/threshold_sweep.csv  (run x task x window: P/R/F1 + TP/FN/FP)
  texs/error_analysis/figures/threshold_sweep.png  (F1 vs window per model; all/pep/propep panels)

Usage:
  env/bin/python analysis/errors/src/threshold_sweep.py [--device 0]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(ROOT))
from analysis.errors.src.error_analysis import run_inference, match_protein, TASKS
from src.utils.manuscript_metrics import convert_path_to_peptide_borders

# Five key models (roadmap #2): baseline, best embedding, the project winner,
# candidate B, and a 3Di (structural) run for the precision-heavy contrast.
MODELS = {
    "train_run_esm2": "ESM2 (baseline)",
    "esmc_6b": "ESM-C 6B",
    "esmc6b_boundary_bond": "ESM-C 6B + boundary/bond ★",
    "esmc6b_telescoping": "ESM-C 6B + telescoping",
    "train_run_esm2+3di_proj_gated_conv": "ESM2 + 3Di (proj gated conv)",
    "esmc6b_3di_gated_boundary": "ESM-C 6B ⊕ 3Di gated + boundary ★new",
}
WINDOWS = [0, 1, 2, 3, 4, 5]


def prf(tp, fn, fp):
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prc = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prc * rec / (prc + rec) if (prc + rec) else 0.0
    return prc, rec, f1


def counts_at_tol(preds, names, data, tol):
    """Aggregate TP/FN/FP over the test split at a given cleavage tolerance."""
    agg = {t: {"tp": 0, "fn": 0, "fp": 0} for t, _, _ in TASKS}
    for i, pred in enumerate(preds):
        row = data.loc[names[i]]
        for task, s_state, e_state in TASKS:
            true_raw = row["true_peptides"] if task == "peptides" else row["true_propeptides"]
            true_segs = [(int(a), int(b)) for a, b in true_raw]
            pred_segs = convert_path_to_peptide_borders(pred, start_state=s_state, stop_state=e_state, offset=1)
            grecs, pred_matched = match_protein(true_segs, pred_segs, tol=tol)
            for gr in grecs:
                agg[task]["tp" if gr["matched"] else "fn"] += 1
            for m in pred_matched:
                if not m:
                    agg[task]["fp"] += 1
    return agg


def sweep_run(run, device):
    preds, names, data, _ = run_inference(ROOT / "runs" / run, device)
    rows = []
    for tol in WINDOWS:
        agg = counts_at_tol(preds, names, data, tol)
        # pooled "all" = peptides + propeptides counts (mirrors compute_all_metrics)
        pooled = {"tp": 0, "fn": 0, "fp": 0}
        for t in ("peptides", "propeptides"):
            for k in pooled:
                pooled[k] += agg[t][k]
        for task, c in [("peptides", agg["peptides"]), ("propeptides", agg["propeptides"]), ("all", pooled)]:
            p, r, f1 = prf(c["tp"], c["fn"], c["fp"])
            rows.append({"run": run, "label": MODELS[run], "task": task, "window": tol,
                         "tp": c["tp"], "fn": c["fn"], "fp": c["fp"],
                         "precision": p, "recall": r, "f1": f1})
    return rows


def reconcile(rows):
    """At tol=3, corrected F1 should reproduce big_metrics_table 'new' F1."""
    bt = ROOT / "analysis" / "metrics" / "big_metrics_table.csv"
    if not bt.exists():
        print("[reconcile] big_metrics_table.csv missing — skipped"); return
    big = {r["run"]: r for r in csv.DictReader(open(bt))}
    print("\n=== reconcile tol=3 corrected F1 vs big_metrics_table new_f1 ===")
    worst = 0.0
    for r in rows:
        if r["window"] != 3:
            continue
        ref = big.get(r["run"], {}).get(f"new_f1_{r['task']}")
        if ref in (None, "", "N/A"):
            continue
        d = abs(r["f1"] - float(ref))
        worst = max(worst, d)
        flag = "  <-- CHECK" if d > 0.01 else ""
        print(f"  {r['run']:38s} {r['task']:11s} sweep={r['f1']:.4f} big={float(ref):.4f} d={d:.4f}{flag}")
    print(f"=== worst |Δ| = {worst:.4f} "
          f"({'OK (uniform small Δ = grouping convention)' if worst < 0.02 else 'LARGE — preds may be wrong'}) ===\n")


def plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    cmap = plt.get_cmap("tab10")
    colors = {run: cmap(i) for i, run in enumerate(MODELS)}
    for ax, task in zip(axes, ["all", "peptides", "propeptides"]):
        for run in MODELS:
            sub = df[(df["run"] == run) & (df["task"] == task)].sort_values("window")
            ax.plot(sub["window"], sub["f1"], marker="o", color=colors[run], label=MODELS[run])
        ax.axvline(3, color="gray", ls=":", lw=1, alpha=.7)
        ax.set_title(task); ax.set_xlabel("acceptance window (±aa on both cleavage sites)")
        ax.set_xticks(WINDOWS); ax.grid(alpha=.3)
    axes[0].set_ylabel("F1 (corrected matcher)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Peptide-finding F1 vs cleavage-tolerance window (test split, fp32)")
    fig.tight_layout()
    out = ROOT / "texs" / "error_analysis" / "figures" / "threshold_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    all_rows = []
    for run in MODELS:
        if not (ROOT / "runs" / run / "model.pt").exists():
            print(f"[skip] {run}: no model.pt"); continue
        print(f"[infer] {run} (fp32)")
        all_rows.extend(sweep_run(run, device))

    df = pd.DataFrame(all_rows)
    out_csv = ROOT / "analysis" / "errors" / "error_stats" / "threshold_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({df['run'].nunique()} runs x {len(WINDOWS)} windows)")
    reconcile(all_rows)
    plot(df)


if __name__ == "__main__":
    main()
