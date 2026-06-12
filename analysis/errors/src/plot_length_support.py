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


def _smooth(a, win=1):
    """Light ±win moving average so the per-integer counts read as a continuous curve."""
    out = np.zeros_like(a, dtype=float)
    for k in range(len(a)):
        lo, hi = max(0, k - win), min(len(a), k + win + 1)
        out[k] = a[lo:hi].mean()
    return out


def main():
    t = pd.read_csv(SEG)
    xs = np.arange(LMIN, LMAX + 1)
    npep = _smooth(counts(t, "peptides"))
    npro = _smooth(counts(t, "propeptides"))
    npep_raw, npro_raw = counts(t, "peptides"), counts(t, "propeptides")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(xs, npep, color="#2b6cb0", alpha=0.35, zorder=2)
    ax.plot(xs, npep, color="#2b6cb0", lw=2, alpha=0.9, label=f"peptides (n={npep_raw.sum()})", zorder=3)
    ax.fill_between(xs, npro, color="#c05621", alpha=0.35, zorder=2)
    ax.plot(xs, npro, color="#c05621", lw=2, alpha=0.9, label=f"propeptides (n={npro_raw.sum()})", zorder=3)
    ax.set_xlim(LMIN, LMAX); ax.set_ylim(0, None)
    ax.set_xlabel("true segment length (aa)")
    ax.set_ylabel("number of true segments (test set, ±1 aa smoothed)")
    ax.set_title("How many true segments per length — support behind the recall-by-length curves")
    ax.grid(alpha=.3)
    ax.legend()
    # mark the thin tails the recall edges rely on
    for lo, hi in [(5, 10), (40, 50)]:
        ax.axvspan(lo, hi, color="grey", alpha=0.07, zorder=1)
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
