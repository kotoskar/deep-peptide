#!/usr/bin/env python3
"""Two error analyses on the project-best esmc6b_boundary_bond (one inference pass):

(1) START-site vs END-site localization error. For each true segment that is DETECTED
    (overlapped by a predicted segment of the SAME task), record signed deltas
    pred_start-true_start and pred_stop-true_stop. If the N-cleavage (start) is much
    worse-localized than the C-cleavage (stop), it empirically motivates symmetrizing the
    telescoping grammar; if symmetric, the grammar rewrite isn't worth the risk. See
    memory deeppeptide-boundary-head-redesign.

(2) PEPTIDE <-> PROPEPTIDE confusion by length. For each true propeptide (stratified by
    length) classify: correct (±3 propep match) / confused (overlapped by a predicted
    PEPTIDE) / missed. Tests the hypothesis that SHORT propeptides are bad because they
    get mis-typed as peptides. Reverse direction (true peptide overlapped by pred propep)
    also reported.

fp32 inference (run_dataloader default). Run from repo root:
  env/bin/python analysis/errors/src/cleavage_localization_and_confusion.py [--device 0]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(ROOT))
from analysis.errors.src.error_analysis import run_inference
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)

RUN = "esmc6b_boundary_bond"
STATS = ROOT / "analysis" / "errors" / "error_stats"
FIG = ROOT / "texs" / "error_analysis" / "figures"


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def best_overlap_pred(true_seg, preds):
    """Among same-task preds overlapping true_seg, the one with max overlap length."""
    best, best_ov = None, 0
    for ps, pe in preds:
        if overlaps(true_seg, (ps, pe)):
            ov = min(true_seg[1], pe) - max(true_seg[0], ps) + 1
            if ov > best_ov:
                best, best_ov = (ps, pe), ov
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    preds, names, data, _ = run_inference(ROOT / "runs" / RUN, device)

    loc_rows = []      # localization deltas (detected true segments)
    conf_rows = []     # per true propeptide / peptide confusion outcome
    for i, pred in enumerate(preds):
        pid = names[i]
        row = data.loc[pid]
        pp = convert_path_to_peptide_borders(pred, start_state=PEPTIDE_START_STATE, stop_state=PEPTIDE_END_STATE, offset=1)
        pr = convert_path_to_peptide_borders(pred, start_state=PROPEPTIDE_START_STATE, stop_state=PROPEPTIDE_END_STATE, offset=1)
        pred_by_task = {"peptides": [(int(a), int(b)) for a, b in pp],
                        "propeptides": [(int(a), int(b)) for a, b in pr]}

        for task in ("peptides", "propeptides"):
            true_segs = [(int(a), int(b)) for a, b in row["true_peptides" if task == "peptides" else "true_propeptides"]]
            other = "propeptides" if task == "peptides" else "peptides"
            for ts in true_segs:
                L = ts[1] - ts[0] + 1
                # (1) localization vs same-task preds
                bp = best_overlap_pred(ts, pred_by_task[task])
                if bp is not None:
                    loc_rows.append({"task": task, "length": L,
                                     "start_err": bp[0] - ts[0], "stop_err": bp[1] - ts[1]})
                # (2) confusion outcome
                correct = any(abs(p[0] - ts[0]) <= 3 and abs(p[1] - ts[1]) <= 3 for p in pred_by_task[task])
                conf_other = any(overlaps(ts, p) for p in pred_by_task[other])
                conf_rows.append({"task": task, "length": L, "correct": correct,
                                  "confused_as_other": conf_other,
                                  "detected_same": best_overlap_pred(ts, pred_by_task[task]) is not None})

    loc = pd.DataFrame(loc_rows)
    conf = pd.DataFrame(conf_rows)
    STATS.mkdir(parents=True, exist_ok=True)
    loc.to_csv(STATS / "cleavage_localization.csv", index=False)
    conf.to_csv(STATS / "task_confusion.csv", index=False)

    # ---- (1) start vs end localization summary ----
    print("=== (1) START vs END localization error (detected true segments) ===")
    for task in ("peptides", "propeptides"):
        t = loc[loc.task == task]
        if not len(t):
            continue
        sa, so = t.start_err.abs(), t.stop_err.abs()
        print(f"\n{task}  (n detected={len(t)})")
        print(f"  |start err|: median {sa.median():.2f}  mean {sa.mean():.2f}  ≤3: {(sa<=3).mean():.3f}  ≤1: {(sa<=1).mean():.3f}")
        print(f"  |stop  err|: median {so.median():.2f}  mean {so.mean():.2f}  ≤3: {(so<=3).mean():.3f}  ≤1: {(so<=1).mean():.3f}")
        print(f"  signed bias start {t.start_err.mean():+.2f}  stop {t.stop_err.mean():+.2f}")

    # ---- (2) propeptide confusion by length ----
    bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 10**9)]
    labels = ["1-5", "6-10", "11-20", "21-30", "31-50", "51+"]
    def lb(L):
        for (lo, hi), lab in zip(bins, labels):
            if lo <= L <= hi:
                return lab
        return "?"
    conf["lenbin"] = conf.length.apply(lb)
    print("\n=== (2) PROPEPTIDE outcome by length (correct ±3 / confused-as-peptide / missed) ===")
    print(f"{'len':>6} {'n':>5} {'correct':>8} {'conf→pep':>9} {'missed':>7}")
    for lab in labels:
        t = conf[(conf.task == "propeptides") & (conf.lenbin == lab)]
        if not len(t):
            continue
        n = len(t)
        correct = t.correct.mean()
        confused = (t.confused_as_other & ~t.correct).mean()
        missed = (~t.detected_same & ~t.confused_as_other).mean()
        print(f"{lab:>6} {n:>5} {correct:>8.3f} {confused:>9.3f} {missed:>7.3f}")
    print("\n--- reverse: PEPTIDE overlapped by a predicted PROPEPTIDE, by length ---")
    print(f"{'len':>6} {'n':>5} {'correct':>8} {'conf→pro':>9}")
    for lab in labels:
        t = conf[(conf.task == "peptides") & (conf.lenbin == lab)]
        if not len(t):
            continue
        print(f"{lab:>6} {len(t):>5} {t.correct.mean():>8.3f} {(t.confused_as_other & ~t.correct).mean():>9.3f}")

    # ---- figures ----
    FIG.mkdir(parents=True, exist_ok=True)
    # fig A: |error| CDF, start vs stop, per task
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, task in zip(axes, ["peptides", "propeptides"]):
        t = loc[loc.task == task]
        for col, lab, c in [("start_err", "start (N-cleavage)", "#2b6cb0"), ("stop_err", "stop (C-cleavage)", "#c05621")]:
            e = np.sort(t[col].abs().values)
            if len(e):
                ax.plot(e, np.arange(1, len(e) + 1) / len(e), label=lab, color=c)
        ax.axvline(3, color="gray", ls=":", lw=1)
        ax.set_title(f"{task} (n={len(t)})"); ax.set_xlabel("|localization error| (aa)")
        ax.set_xlim(0, 15); ax.grid(alpha=.3); ax.legend()
    axes[0].set_ylabel("cumulative fraction of detected segments")
    fig.suptitle(f"Cleavage-site localization: start vs end — {RUN}")
    fig.tight_layout(); fig.savefig(FIG / "cleavage_localization.png", dpi=130); plt.close(fig)

    # fig B: propeptide outcome stacked bars by length
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs, cor, con, mis = [], [], [], []
    for lab in labels:
        t = conf[(conf.task == "propeptides") & (conf.lenbin == lab)]
        if not len(t):
            continue
        xs.append(f"{lab}\n(n={len(t)})")
        cor.append(t.correct.mean())
        con.append((t.confused_as_other & ~t.correct).mean())
        mis.append((~t.detected_same & ~t.confused_as_other).mean())
    x = np.arange(len(xs))
    ax.bar(x, cor, label="correct (±3 propep)", color="#2f855a")
    ax.bar(x, con, bottom=cor, label="confused → peptide", color="#c53030")
    ax.bar(x, mis, bottom=np.array(cor) + np.array(con), label="missed (no overlap)", color="#a0aec0")
    ax.set_xticks(x); ax.set_xticklabels(xs); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of true propeptides"); ax.set_xlabel("true propeptide length (aa)")
    ax.set_title(f"Propeptide outcome by length — {RUN}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "propeptide_confusion_by_length.png", dpi=130); plt.close(fig)
    print(f"\nwrote {FIG/'cleavage_localization.png'} , {FIG/'propeptide_confusion_by_length.png'}")
    print(f"wrote {STATS/'cleavage_localization.csv'} , {STATS/'task_confusion.csv'}")


if __name__ == "__main__":
    main()
