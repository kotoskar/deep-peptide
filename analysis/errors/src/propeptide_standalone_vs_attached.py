#!/usr/bin/env python3
"""Standalone vs peptide-attached propeptides: composition-by-length figure + recall.

A true propeptide is "attached" if its protein also contains an annotated PEPTIDE
(verified earlier: within such proteins ~all propeptides are directly adjacent, ≤5 aa,
to a peptide), else "standalone". This explains the short-propeptide failure: short
propeptides are overwhelmingly STANDALONE, and standalone short ones are the hardest.

Outputs (TEST set = cluster 4, the winner esmc6b_boundary_bond):
  - texs/error_analysis/figures/propeptide_standalone_fraction_by_length.png
    (stacked area: fraction standalone vs attached by length, per-integer, ±2 smoothed)
  - prints the two headline recall numbers (standalone vs attached) + recall by length.

Run from repo root:
  env/bin/python analysis/errors/src/propeptide_standalone_vs_attached.py
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
SEG = ROOT / "analysis" / "errors" / "error_stats" / "esmc6b_boundary_bond__segments.csv"
LAB = ROOT / "data" / "uniprot_2022" / "labeled_sequences.csv"
FIG = ROOT / "texs" / "error_analysis" / "figures"
LMIN, LMAX, WIN = 5, 50, 2


def main():
    seg = pd.read_csv(SEG)
    lab = pd.read_csv(LAB)
    has_pep = {r.protein_id: bool(re.search(r"\(\d+-\d+\)", str(r.coordinates)))
               for r in lab.itertuples()}

    t = seg[(seg.task == "propeptides") & (seg.kind == "true")].copy()
    t["attached"] = t.protein_id.map(has_pep).fillna(False)
    t["cat"] = np.where(t["attached"], "attached", "standalone")

    # ---- headline recall numbers ----
    rec = t.groupby("cat").matched.agg(n="size", recall="mean")
    print("=== Recall (±3) by propeptide category — winner, TEST set ===")
    print(rec.round(3).to_string())
    print(f"\nTWO NUMBERS: recall standalone = {t[t.cat=='standalone'].matched.mean():.3f}  "
          f"| recall attached = {t[t.cat=='attached'].matched.mean():.3f}")

    # ---- recall by length x category (context) ----
    def lb(L): return "6-10" if 6 <= L <= 10 else "11-20" if 11 <= L <= 20 else "21-30" if 21 <= L <= 30 else "31-50"
    t["lb"] = t.length.apply(lb)
    print("\nrecall by length x category:")
    print(t.pivot_table(index="lb", columns="cat", values="matched", aggfunc=["mean", "size"])
            .round(3).reindex(["6-10", "11-20", "21-30", "31-50"]).to_string())

    # ---- composition-by-length figure (fraction standalone vs attached) ----
    counts_s = np.zeros(LMAX - LMIN + 1)
    counts_a = np.zeros(LMAX - LMIN + 1)
    for L in range(LMIN, LMAX + 1):
        sub = t[t.length == L]
        counts_s[L - LMIN] = (sub.cat == "standalone").sum()
        counts_a[L - LMIN] = (sub.cat == "attached").sum()
    xs = np.arange(LMIN, LMAX + 1)
    frac_s = np.zeros_like(xs, dtype=float)
    ntot = np.zeros_like(xs, dtype=float)
    for k in range(len(xs)):
        lo, hi = max(0, k - WIN), min(len(xs), k + WIN + 1)
        s, a = counts_s[lo:hi].sum(), counts_a[lo:hi].sum()
        ntot[k] = s + a
        frac_s[k] = s / (s + a) if (s + a) else np.nan
    frac_a = 1 - frac_s

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(xs, frac_s, frac_a, colors=["#2b6cb0", "#dd9b3b"],
                 labels=["standalone (protein has NO peptide)",
                         "attached (protein has a peptide; ~all adjacent ≤5aa)"], alpha=0.9)
    ax.plot(xs, frac_s, color="white", lw=1, alpha=.7)
    ax.set_xlim(LMIN, LMAX); ax.set_ylim(0, 1)
    ax.set_xlabel("true propeptide length (aa)")
    ax.set_ylabel("fraction of true propeptides (±2 aa smoothed)")
    ax.set_title("Propeptide composition by length: standalone vs peptide-attached (test set)")
    ax.legend(loc="lower center", fontsize=9, framealpha=.9)
    # annotate raw count at a few lengths
    for L in (6, 10, 20, 30, 45):
        k = L - LMIN
        ax.annotate(f"n≈{int(ntot[k])}", (L, 0.04), ha="center", fontsize=7, color="white")
    fig.tight_layout()
    out = FIG / "propeptide_standalone_fraction_by_length.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
