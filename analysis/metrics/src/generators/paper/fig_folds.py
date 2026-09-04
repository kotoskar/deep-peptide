#!/usr/bin/env python3
"""Appendix figure: how the seven GraphPart folds of the earlier single-split
protocol differ in segment-length composition.

Panel (a) draws the segment-length profile of every fold (nine 5-residue bins
from 5-9 to 45-50, as a share of that fold's segments) against the profile of
the whole dataset, which is the heavy INK line. Each fold carries its own
colour *and* marker *and* dash (`FOLD_STYLE`), so the folds the text and the
caption name -- 2, 4 and 5 -- can actually be found in the tangle. Panel (b)
reduces each fold to one number, the L1 distance between its profile and the
overall profile, and sorts the folds by it.

Ported from the `fold_divergence` block of
`analysis/metrics/src/generators/report_figs_data.py` (the Russian report
generator). The data logic there -- the coordinate parser, the fold merge, the
bin edges, the row-wise accumulation, the per-fold normalisation and the L1 --
is carried over unchanged, so the numbers are identical; only labels, colours,
fonts, sizing and layout are new. That generator is untouched and keeps writing
the Russian figure.

Input : data/uniprot_2026/labeled_sequences.csv     (segment coordinates)
        data/uniprot_2026/graphpart_assignments.csv (protein -> fold)
Output: texs/ai4dd/figures/fold_divergence.png

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_folds.py
"""
from __future__ import annotations

import re
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (AIRI, INK, MUTE, apply, textwidth, panel,  # noqa: E402
                     finish)

import matplotlib.pyplot as plt                                  # noqa: E402


# ------------------------------------------------------------------- data ---
# Verbatim from report_figs_data.py (lines 18, 21, 49-65).

def segs(s):
    return ([(int(a), int(b)) for a, b in re.findall(r"\((\d+)-(\d+)\)", s)]
            if isinstance(s, str) else [])


def load():
    d26 = pd.read_csv("data/uniprot_2026/labeled_sequences.csv")
    asg = (pd.read_csv("data/uniprot_2026/graphpart_assignments.csv")
             .rename(columns={"AC": "protein_id", "cluster": "fold"}))
    df = d26.merge(asg, on="protein_id")

    LBINS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]
    keys = [f"{LBINS[i]}-{LBINS[i + 1] - 1}" for i in range(len(LBINS) - 1)]

    def lbin(L):
        for i in range(len(LBINS) - 1):
            if LBINS[i] <= L < LBINS[i + 1]:
                return keys[i]
        return None

    rows = []
    for _, r in df.iterrows():
        for col in ("coordinates", "propeptide_coordinates"):
            for a, b in segs(r.get(col)):
                lb = lbin(b - a + 1)
                if lb:
                    rows.append((int(r["fold"]), lb))
    S = pd.DataFrame(rows, columns=["fold", "lb"])
    piv = S.groupby(["fold", "lb"]).size().unstack(fill_value=0)
    piv = piv.div(piv.sum(1), axis=0)[keys]
    glob = (S.groupby("lb").size() / len(S))[keys]
    L1 = {f: float(np.abs(piv.loc[f].values - glob.values).sum()) for f in piv.index}
    return keys, piv, glob, L1


# ---------------------------------------------------------------- drawing ---
# One (colour, marker, dash, linewidth) per fold, in the `STYLE` pattern of
# `_common.py`: colour is never the only channel, so the folds stay separable
# in greyscale and for colour-vision deficiencies. Local to this figure -- no
# other paper figure draws folds.
#
# The hues are deliberately disjoint from panel (b)'s bar colours (AIRI A / E21
# / E3) and from the INK of the whole-dataset line, so a reader carrying the
# panel-(a) key into (b) is not misled. They also keep the segment-*type* key
# (`_common.TYPE_COLOR`: peptides = E1 orange, propeptides = B teal) off the two
# emphasised lines: the dataset-composition figure draws exactly those two hues
# on exactly this x-axis, and a heavy orange and a heavy teal curve here would
# read as that pair. E1 and B survive only as thin dashed lines.
#
# Folds 2, 4 and 5 are the ones the text and the caption name, so they get the
# three most separable hues and a heavier line; the other four stay thin. No two
# folds share a dash pattern, so the figure survives a greyscale print.
FOLD_STYLE = {
    0: (AIRI["F2"], "^", (0, (4, 1.2, 1, 1.2)),      0.9),  # violet, dash-dot
    1: (AIRI["D1"], "P", (0, (1.2, 1.2)),            1.1),  # tan, dotted
    2: (AIRI["F3"], "D", "-",                        1.5),  # magenta, solid   (text)
    3: (AIRI["B"],  "v", (0, (3, 1, 1, 1, 1, 1)),    0.9),  # teal, dash-dot-dot
    4: (AIRI["F1"], "s", (0, (6, 1.8)),              1.5),  # blue, long dash  (caption)
    5: (AIRI["C"],  "X", (0, (2, 1.2)),              1.5),  # brown, short dash (text)
    6: (AIRI["E1"], "*", (0, (5, 1.5, 1.5, 1.5)),    0.9),  # orange, long-short
}


def draw_profiles(ax, keys, piv, glob):
    """One styled line per fold, the whole dataset heavy on top."""
    x = np.arange(len(keys))

    for f in piv.index:
        col, mk, ls, lw = FOLD_STYLE[f]
        ax.plot(x, piv.loc[f].values * 100, color=col, marker=mk, linestyle=ls,
                lw=lw, ms=3.6 if mk == "*" else 2.7, mew=0,
                label=f"fold {f}", zorder=3 if lw > 1.2 else 2)
    ax.plot(x, glob.values * 100, "o-", color=INK, lw=1.8, ms=3.2,
            label="whole dataset", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=0, fontsize=7)
    ax.set_xlabel("segment length")
    ax.set_ylabel("share of segments, %")
    ax.set_ylim(0, 36)
    ax.grid(axis="x", alpha=0)

    # Whole dataset first, then fold 0 .. fold 6. The legend sits over the
    # gridlines, so it needs an opaque white patch: `paperstyle` turns frames
    # off globally and the y gridlines would otherwise strike through the text.
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("whole dataset")] + \
            [i for i, lab in enumerate(labels) if lab != "whole dataset"]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              ncol=2, loc="upper right", handlelength=2.4,
              # borderaxespad=0 so the opaque patch reaches the right spine: any
              # inset leaves the gridlines showing as orphaned slivers beside it.
              columnspacing=1.0, labelspacing=0.32, borderaxespad=0.0,
              frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="none", borderpad=0.35)


def draw_l1(ax, L1):
    """One bar per fold, ascending; colour flags the best fold and the >0.23 tail."""
    order = sorted(L1, key=lambda f: L1[f])          # full precision, as in the original
    yy = np.arange(len(order))[::-1]
    cols = [AIRI["A"] if f == order[0]
            else AIRI["E21"] if L1[f] > 0.23
            else AIRI["E3"] for f in order]

    ax.barh(yy, [L1[f] for f in order], color=cols, edgecolor="white",
            linewidth=0.6, height=0.72)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"fold {f}" for f in order])
    for yi, f in zip(yy, order):
        ax.text(L1[f] + 0.006, yi, f"{L1[f]:.2f}", va="center",
                fontsize=6.5, color=MUTE,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.22))
    ax.set_xlim(0, max(L1.values()) * 1.18)
    ax.set_xlabel("deviation from overall profile (L1)")
    ax.grid(axis="y", alpha=0)


# ------------------------------------------------------------------- main ---

def main() -> int:
    keys, piv, glob, L1 = load()

    apply()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(textwidth(0.7), 4.85),
        gridspec_kw={"height_ratios": [1.45, 1.0]}, layout="constrained")

    draw_profiles(ax1, keys, piv, glob)
    draw_l1(ax2, L1)
    panel(ax1, "a", "Length profile by fold", dx=-0.12)
    panel(ax2, "b", "Fold length-profile deviation from overall", dx=-0.12)

    finish(fig, "fold_divergence")
    print("L1:", {f: round(v, 3) for f, v in sorted(L1.items())})
    print("whole dataset, %:", [round(v * 100, 1) for v in glob.values])
    for f in piv.index:
        print(f"fold {f}, %:", [round(v * 100, 1) for v in piv.loc[f].values])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
