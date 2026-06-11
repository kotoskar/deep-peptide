#!/usr/bin/env python3
"""Stacked-area outcome-by-length plots for peptide<->propeptide confusion.

Reads analysis/errors/error_stats/task_confusion.csv (from
cleavage_localization_and_confusion.py — no re-inference needed) and draws, for each
task, a stacked area over true length: per-residue-length fraction of true segments that
are correct / near-miss(same type) / confused(other type) / missed.

Mutually-exclusive priority per true segment (so bands sum to 1):
  correct (±3 same-task)  >  near-miss (same-task overlap, not ±3)
  >  confused (only an OTHER-task pred overlaps)  >  missed (nothing overlaps).
Per-integer-length fractions are smoothed with a centered ±2 rolling window (counts
summed then normalised) to tame small-sample noise (some lengths have <20 segments).

Run from repo root:
  env/bin/python analysis/errors/src/plot_confusion_stacked.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
CSV = ROOT / "analysis" / "errors" / "error_stats" / "task_confusion.csv"
FIG = ROOT / "texs" / "error_analysis" / "figures"

LMIN, LMAX, WIN = 5, 50, 2          # length range and ±WIN smoothing window
CATS = ["correct", "nearmiss", "confused", "missed"]
COLORS = {"correct": "#2f855a", "nearmiss": "#d69e2e", "confused": "#c53030", "missed": "#a0b0c0"}


def outcome(r):
    if r["correct"]:
        return "correct"
    if r["detected_same"]:
        return "nearmiss"
    if r["confused_as_other"]:
        return "confused"
    return "missed"


def smoothed_fractions(t):
    """Return x lengths and a dict cat -> smoothed fraction array."""
    t = t.copy()
    t["outcome"] = t.apply(outcome, axis=1)
    # raw counts per integer length x category
    counts = {c: np.zeros(LMAX - LMIN + 1) for c in CATS}
    for L in range(LMIN, LMAX + 1):
        sub = t[t.length == L]
        for c in CATS:
            counts[c][L - LMIN] = (sub["outcome"] == c).sum()
    xs = np.arange(LMIN, LMAX + 1)
    # centered rolling sum (±WIN), then normalise to fractions
    fr = {c: np.zeros_like(xs, dtype=float) for c in CATS}
    for k, L in enumerate(xs):
        lo, hi = max(0, k - WIN), min(len(xs), k + WIN + 1)
        tot = sum(counts[c][lo:hi].sum() for c in CATS)
        if tot > 0:
            for c in CATS:
                fr[c][k] = counts[c][lo:hi].sum() / tot
    return xs, fr, {L: int(sum(counts[c][L - LMIN] for c in CATS)) for L in xs}


def main():
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    panels = [("propeptides", "→ peptide", axes[0]), ("peptides", "→ propeptide", axes[1])]
    for task, conf_lbl, ax in panels:
        t = df[df.task == task]
        xs, fr, ntot = smoothed_fractions(t)
        labels = {"correct": "correct (±3)", "nearmiss": "near-miss (same type)",
                  "confused": f"confused ({conf_lbl})", "missed": "missed (no overlap)"}
        ax.stackplot(xs, [fr[c] for c in CATS],
                     colors=[COLORS[c] for c in CATS],
                     labels=[labels[c] for c in CATS], alpha=0.92)
        # crisp boundary lines between bands
        cum = np.zeros_like(xs, dtype=float)
        for c in CATS[:-1]:
            cum = cum + fr[c]
            ax.plot(xs, cum, color="white", lw=1.0, alpha=0.8)
        ax.set_xlim(LMIN, LMAX); ax.set_ylim(0, 1)
        ax.set_xlabel(f"true {task[:-1]} length (aa)")
        ax.set_title(f"{task} (n={len(t)})")
        ax.legend(loc="lower center", fontsize=8, framealpha=0.9, ncol=2)
    axes[0].set_ylabel("fraction of true segments (±2 aa smoothed)")
    fig.suptitle("Outcome by length — esmc6b_boundary_bond  (peptide ↔ propeptide confusion)")
    fig.tight_layout()
    out = FIG / "confusion_by_length_stacked.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
