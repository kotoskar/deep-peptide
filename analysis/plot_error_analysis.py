#!/usr/bin/env python3
"""Plot the error-analysis breakdowns (length / organism) into texs/error_analysis/figures/.

Reads analysis/error_stats/all_segments.csv (produced by error_analysis.py) and
emits PNG figures + does not write the report (report.md is authored separately).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("texs/error_analysis/figures")
FIG.mkdir(parents=True, exist_ok=True)
LEN_LABELS = ["5", "6-10", "11-20", "21-30", "31-50"]
BINS = {"5": (5, 5), "6-10": (6, 10), "11-20": (11, 20), "21-30": (21, 30), "31-50": (31, 50)}


def lenbin(L):
    for lab, (lo, hi) in BINS.items():
        if lo <= L <= hi:
            return lab
    return None


def main():
    df = pd.read_csv("analysis/error_stats/all_segments.csv")
    df = df[df.get("gate_ok", True)] if "gate_ok" in df.columns else df
    true = df[df["kind"] == "true"].copy()
    true["lenbin"] = true["length"].apply(lenbin)

    # ---- Fig 1: recall by length bin (pep + propep) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(LEN_LABELS)); w = 0.38
    for off, task, color in [(-w/2, "peptides", "#3b78c2"), (w/2, "propeptides", "#c2683b")]:
        t = true[true["task"] == task]
        recs = [t[t["lenbin"] == lab]["matched"].mean() if len(t[t["lenbin"] == lab]) else np.nan for lab in LEN_LABELS]
        ax.bar(x + off, recs, w, label=task, color=color)
    ax.set_xticks(x); ax.set_xticklabels(LEN_LABELS)
    ax.set_xlabel("true peptide length (aa)"); ax.set_ylabel("recall (±3 matching)")
    ax.set_title("Recall by peptide length"); ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "recall_by_length.png", dpi=130); plt.close(fig)

    # ---- Fig 2: FN mass (count of misses) by length bin ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for off, task, color in [(-w/2, "peptides", "#3b78c2"), (w/2, "propeptides", "#c2683b")]:
        t = true[true["task"] == task]
        fns = [int((~t[t["lenbin"] == lab]["matched"]).sum()) for lab in LEN_LABELS]
        ax.bar(x + off, fns, w, label=task, color=color)
    ax.set_xticks(x); ax.set_xticklabels(LEN_LABELS)
    ax.set_xlabel("true peptide length (aa)"); ax.set_ylabel("# false negatives (missed)")
    ax.set_title("Where the missed peptides are (FN count by length)"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fn_mass_by_length.png", dpi=130); plt.close(fig)

    # ---- Fig 3: recall by organism (peptides, top 12 by count) ----
    t = true[true["task"] == "peptides"]
    top = t["organism"].value_counts().head(12)
    org_rec = [(org, t[t["organism"] == org]["matched"].mean(), n) for org, n in top.items()]
    org_rec.sort(key=lambda z: z[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ys = np.arange(len(org_rec))
    colors = ["#b5402f" if r < 0.4 else ("#d9a441" if r < 0.6 else "#3f8f4f") for _, r, _ in org_rec]
    ax.barh(ys, [r for _, r, _ in org_rec], color=colors)
    ax.set_yticks(ys); ax.set_yticklabels([f"{o}  (n={n})" for o, _, n in org_rec], fontsize=8)
    ax.set_xlabel("recall (±3 matching)"); ax.set_xlim(0, 1)
    ax.set_title("Recall by organism — peptides (top 12 by count)")
    ax.axvline(0.4, color="grey", ls="--", lw=.8); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "recall_by_organism.png", dpi=130); plt.close(fig)

    # ---- Fig 4: per-run recall (pep/propep) ----
    runs = sorted(true["run"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xr = np.arange(len(runs))
    for off, task, color in [(-w/2, "peptides", "#3b78c2"), (w/2, "propeptides", "#c2683b")]:
        recs = [true[(true["run"] == r) & (true["task"] == task)]["matched"].mean() for r in runs]
        ax.bar(xr + off, recs, w, label=task, color=color)
    ax.set_xticks(xr); ax.set_xticklabels([r.replace("train_run_", "").replace("esm2_", "") for r in runs],
                                          rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("recall (±3, corrected)"); ax.set_title("Per-run recall (corrected ±3 matching)")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "recall_by_run.png", dpi=130); plt.close(fig)

    print("wrote figures to", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
