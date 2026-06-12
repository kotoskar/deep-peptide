#!/usr/bin/env python3
"""Support figure for the recall-by-length plots: count of TRUE peptides / propeptides
at each integer length (test set). Ground truth is model-independent, so this shows how
many segments back each point of the recall-by-length curves — i.e. where the tails are
statistically thin and recall differences are noisy.

Reads true segments from a segments csv (default the winner's, full 1533-protein test).

Run from repo root:
  env/bin/python analysis/errors/src/plot_length_support.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
SEG = ROOT / "analysis" / "errors" / "error_stats" / "esmc6b_boundary_bond__segments.csv"
FIG = ROOT / "texs" / "error_analysis" / "figures"
LMIN, LMAX = 5, 50


def counts(t, task):
    s = t[(t.kind == "true") & (t.task == task)]
    return np.array([(s.length == L).sum() for L in range(LMIN, LMAX + 1)])


def main():
    t = pd.read_csv(SEG)
    xs = np.arange(LMIN, LMAX + 1)
    npep = counts(t, "peptides")
    npro = counts(t, "propeptides")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.42
    ax.bar(xs - w / 2, npep, w, color="#2b6cb0", label=f"peptides (n={npep.sum()})")
    ax.bar(xs + w / 2, npro, w, color="#c05621", label=f"propeptides (n={npro.sum()})")
    ax.set_xlim(LMIN - 0.5, LMAX + 0.5)
    ax.set_xlabel("true segment length (aa)")
    ax.set_ylabel("number of true segments (test set)")
    ax.set_title("How many true segments per length — support behind the recall-by-length curves")
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    # mark the thin tails the recall edges rely on
    for lo, hi, txt in [(5, 10, "short tail"), (40, 50, "long tail")]:
        ax.axvspan(lo - 0.5, hi + 0.5, color="grey", alpha=0.07)
    fig.tight_layout()
    out = FIG / "length_support_counts.png"
    fig.savefig(out, dpi=140); plt.close(fig)

    # quick console summary of tail thinness
    def band(a, lo, hi):
        return int(a[(xs >= lo) & (xs <= hi)].sum())
    print("peptides   5-10:", band(npep, 5, 10), "| 11-30:", band(npep, 11, 30), "| 31-50:", band(npep, 31, 50))
    print("propeptides 5-10:", band(npro, 5, 10), "| 11-30:", band(npro, 11, 30), "| 31-50:", band(npro, 31, 50))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
