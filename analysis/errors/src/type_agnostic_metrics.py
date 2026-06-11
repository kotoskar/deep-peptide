#!/usr/bin/env python3
"""Type-agnostic ±3 metric ("any") vs the typed metric, per model.

For every infer-able run we decode predicted peptide & propeptide borders, then score
two ways with the SAME corrected ±3 matcher:
  - TYPED ("all"): true peptides matched only by predicted PEPTIDES, true propeptides
    only by predicted PROPEPTIDES, counts pooled. (== big_metrics_table 'new' all.)
  - ANY: merge both true types into one set and both predicted types into one set;
    a true segment is a TP iff SOME prediction (EITHER type) lands within ±3 of both
    cleavage sites. Type is ignored.
F1_any >= F1_all always; the GAP (F1_any - F1_all) is exactly the cost of peptide<->
propeptide TYPE confusion among otherwise-correctly-localized segments.

Also reports the confusion fraction in both directions: of true segments that ARE
localized (some pred within ±3, either type), what fraction are mis-typed (only the
WRONG-type prediction matches).

fp32 inference. Reconcile: typed F1_all must reproduce big_metrics_table new_f1_all.

Run from repo root:
  env/bin/python analysis/errors/src/type_agnostic_metrics.py [--only r1 r2 ...] [--device 0]
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
from analysis.errors.src.error_analysis import run_inference, match_protein
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)

STATS = ROOT / "analysis" / "errors" / "error_stats"
FIG = ROOT / "texs" / "error_analysis" / "figures"
SKIP = {"esm2_bond_loss_soft_l005_w5_tau15", "esm2_aho_transition_bias_sparse_trainable_zero"}


def prf(tp, fn, fp):
    r = tp / (tp + fn) if (tp + fn) else 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def within3(a, segs):
    return any(abs(s[0] - a[0]) <= 3 and abs(s[1] - a[1]) <= 3 for s in segs)


def score_run(run, device):
    preds, names, data, _ = run_inference(ROOT / "runs" / run, device)
    # typed pooled counts, any counts, confusion counts
    typed = {"tp": 0, "fn": 0, "fp": 0}
    anyc = {"tp": 0, "fn": 0, "fp": 0}
    # confusion: true segments that are localized (some pred±3, either type) but mistyped
    conf = {"peptides": {"loc": 0, "mistyped": 0}, "propeptides": {"loc": 0, "mistyped": 0}}
    for i, pred in enumerate(preds):
        row = data.loc[names[i]]
        tp_pep = [(int(a), int(b)) for a, b in row["true_peptides"]]
        tp_pro = [(int(a), int(b)) for a, b in row["true_propeptides"]]
        pr_pep = [(int(a), int(b)) for a, b in convert_path_to_peptide_borders(pred, PEPTIDE_START_STATE, PEPTIDE_END_STATE, 1)]
        pr_pro = [(int(a), int(b)) for a, b in convert_path_to_peptide_borders(pred, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE, 1)]

        # TYPED: per task, pooled
        for tt, pp in [(tp_pep, pr_pep), (tp_pro, pr_pro)]:
            g, pm = match_protein(tt, pp, tol=3)
            for gr in g:
                typed["tp" if gr["matched"] else "fn"] += 1
            typed["fp"] += sum(1 for m in pm if not m)
        # ANY: merge types
        g, pm = match_protein(tp_pep + tp_pro, pr_pep + pr_pro, tol=3)
        for gr in g:
            anyc["tp" if gr["matched"] else "fn"] += 1
        anyc["fp"] += sum(1 for m in pm if not m)
        # CONFUSION per true segment (located = within3 of either type; mistyped = only other type)
        for task, trues, same, other in [("peptides", tp_pep, pr_pep, pr_pro),
                                          ("propeptides", tp_pro, pr_pro, pr_pep)]:
            for ts in trues:
                in_same, in_other = within3(ts, same), within3(ts, other)
                if in_same or in_other:
                    conf[task]["loc"] += 1
                    if in_other and not in_same:
                        conf[task]["mistyped"] += 1
    pt = prf(**typed)
    pa = prf(**anyc)
    loc_tot = conf["peptides"]["loc"] + conf["propeptides"]["loc"]
    mis_tot = conf["peptides"]["mistyped"] + conf["propeptides"]["mistyped"]
    return {
        "run": run,
        "p_all": pt[0], "r_all": pt[1], "f1_all": pt[2],
        "p_any": pa[0], "r_any": pa[1], "f1_any": pa[2],
        "f1_gap": pa[2] - pt[2],
        "mistyped_frac": mis_tot / loc_tot if loc_tot else 0.0,
        "pep_mistyped_frac": conf["peptides"]["mistyped"] / conf["peptides"]["loc"] if conf["peptides"]["loc"] else 0.0,
        "propep_mistyped_frac": conf["propeptides"]["mistyped"] / conf["propeptides"]["loc"] if conf["propeptides"]["loc"] else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    runs = args.only or sorted(d.name for d in (ROOT / "runs").iterdir()
                               if (d / "model.pt").exists() and (d / "config.json").exists()
                               and d.name not in SKIP)
    rows = []
    for run in runs:
        try:
            rows.append(score_run(run, device))
            print(f"[ok] {run:42s} F1_all={rows[-1]['f1_all']:.3f} F1_any={rows[-1]['f1_any']:.3f} "
                  f"gap={rows[-1]['f1_gap']:+.3f} mistyped={rows[-1]['mistyped_frac']:.3f}")
        except Exception as e:
            print(f"[FAIL] {run}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    out = STATS / "type_agnostic_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} runs)")

    # reconcile typed vs big_table
    bt = ROOT / "analysis" / "metrics" / "big_metrics_table.csv"
    if bt.exists():
        big = {r["run"]: r for r in csv.DictReader(open(bt))}
        worst = 0.0
        for r in rows:
            ref = big.get(r["run"], {}).get("new_f1_all")
            if ref not in (None, "", "N/A"):
                worst = max(worst, abs(r["f1_all"] - float(ref)))
        print(f"reconcile typed F1_all vs big_table new_f1_all: worst |Δ|={worst:.4f}")

    # figure: F1_all vs F1_any per run (sorted by gap), only if many runs
    if len(df) >= 4:
        d = df.sort_values("f1_gap", ascending=True)
        y = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(d))))
        ax.hlines(y, d["f1_all"], d["f1_any"], color="#cbd5e0", lw=2, zorder=1)
        ax.scatter(d["f1_all"], y, color="#2b6cb0", label="F1 typed (all)", zorder=2)
        ax.scatter(d["f1_any"], y, color="#c05621", label="F1 any (type-agnostic)", zorder=2)
        ax.set_yticks(y); ax.set_yticklabels(d["run"], fontsize=7)
        ax.set_xlabel("F1 (±3)"); ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Typed vs type-agnostic F1 — gap = peptide↔propeptide confusion")
        ax.grid(axis="x", alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "type_agnostic_f1.png", dpi=140); plt.close(fig)
        print(f"wrote {FIG/'type_agnostic_f1.png'}")


if __name__ == "__main__":
    main()
