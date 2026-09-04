#!/usr/bin/env python3
"""Appendix figure: what the 2026 dataset looks like -- segment lengths and hosts.

Panel (a) overlays the length histograms of the annotated mature peptides and of
the propeptides, so the reader can see that the two classes cover the same 5--50
residue band and that neither is a long-tail curiosity. Panel (b) lists the ten
genera that contribute the most precursor proteins, which is where the dataset's
taxonomic imbalance -- cone snails first, then the usual model organisms -- is
visible at a glance.

Ported from the Russian report generator
``analysis/metrics/src/generators/report_figs_data.py`` (its first block, the one
that writes ``data_distributions.png``). The data logic is carried over verbatim:
same CSV, same coordinate regex, same ``np.arange(5, 52, 3)`` bins, same
``organism -> first whitespace-delimited token`` genus rule and same top-10
``value_counts``. Only the language, the palette, the type sizes and the print
size differ, so the numbers are identical to the Russian figure.

Input : data/uniprot_2026/labeled_sequences.csv
Output: texs/ai4dd/figures/data_distributions.png

main.tex includes this at 0.9\\linewidth, hence ``textwidth(0.9)``: drawn at its
true print size the PNG lands at scale 1 and the 8 pt type stays 8 pt on paper.

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_datadist.py
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

from _common import (AIRI, MUTE, TYPE_COLOR, apply,       # noqa: E402
                     finish, panel, textwidth)

CSV = "data/uniprot_2026/labeled_sequences.csv"

# English common names for the genera that reach the top 10 (the Russian
# generator carries the same mapping in Cyrillic). Kept a couple of entries
# longer than the top 10 so the figure survives a dataset refresh.
COMMON = {
    "Conus": "cone snails",
    "Mus": "mice",
    "Homo": "human",
    "Cyriopagopus": "tarantulas",
    "Rattus": "rats",
    "Arabidopsis": "thale cress",
    "Bos": "cattle",
    "Lycosa": "wolf spiders",
    "Saccharomyces": "yeast",
    "Drosophila": "fruit flies",
    "Gallus": "chickens",
    "Sus": "pigs",
}

# The bars carry no series meaning -- one neutral brand colour, dark enough that
# the grey value labels beside it stay legible. Deliberately not a segment-type
# hue, so panel (b) cannot be read as a continuation of panel (a)'s legend.
BAR_COLOR = AIRI["A"]

_SEG_RE = re.compile(r"\((\d+)-(\d+)\)")


def segs(s):
    """Parse a '(start-end),(start-end)' coordinate string, as in the original."""
    return [(int(a), int(b)) for a, b in _SEG_RE.findall(s)] if isinstance(s, str) else []


def main() -> int:
    d26 = pd.read_csv(CSV)
    pep = [b - a + 1 for r in d26["coordinates"] for a, b in segs(r)]
    pro = [b - a + 1 for r in d26["propeptide_coordinates"] for a, b in segs(r)]
    d26["genus"] = d26.organism.fillna("?").str.split().str[0]
    gtop = d26.genus.value_counts().head(10)

    apply()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(textwidth(0.9), 3.25),
        gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")

    # ------------------------------------------------ (a) segment lengths ---
    bins = np.arange(5, 52, 3)
    ax1.hist(pep, bins=bins, color=TYPE_COLOR["peptides"], alpha=0.70,
             edgecolor="white", linewidth=0.5, label=f"peptides (n={len(pep)})")
    ax1.hist(pro, bins=bins, color=TYPE_COLOR["propeptides"], alpha=0.55,
             edgecolor="white", linewidth=0.5, label=f"propeptides (n={len(pro)})")
    ax1.set_xlabel("segment length (residues)")
    ax1.set_ylabel("number of segments")
    ax1.set_xlim(bins[0] - 1, bins[-1] + 1)
    tallest = max(np.histogram(pep, bins=bins)[0].max(),
                  np.histogram(pro, bins=bins)[0].max())
    # Headroom for the legend, so it never sits on top of a bar.
    ax1.set_ylim(0, tallest * 1.24)
    # ... and keep the gridlines inside the bar range: the headroom band above
    # the tallest bar belongs to the legend, and a gridline drawn up there runs
    # straight through the descenders of its second row.
    ax1.set_yticks([t for t in ax1.get_yticks() if 0 <= t <= tallest])
    ax1.grid(axis="x", alpha=0)          # length axis is continuous, y grid only
    ax1.legend(loc="upper right", handlelength=1.4, handletextpad=0.5,
               borderaxespad=0.2, labelspacing=0.35)
    panel(ax1, "a", "Length distribution", dx=-0.135)

    # -------------------------------------------------- (b) host organisms ---
    y = np.arange(len(gtop))[::-1]
    ax2.barh(y, gtop.values, height=0.74, color=BAR_COLOR,
             edgecolor="white", linewidth=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{g}\n({COMMON.get(g, '?')})" for g in gtop.index],
                        fontsize=6.8, linespacing=1.15)
    for yi, v in zip(y, gtop.values):
        ax2.text(v + gtop.values.max() * 0.025, yi, str(v), va="center",
                 fontsize=6.5, color=MUTE)
    ax2.set_xlabel("number of proteins")
    ax2.set_xlim(0, gtop.values.max() * 1.16)
    ax2.set_ylim(y.min() - 0.65, y.max() + 0.65)
    # No grid in this panel. Every bar is annotated with its exact count, so a
    # vertical gridline adds no information -- and because the grid sits behind
    # the bars it would only ever be visible in the whitespace the value labels
    # occupy, striking through "921", "461"/"457" and "217"/"175"/"166".
    ax2.grid(False)
    panel(ax2, "b", "Most frequent genera", dx=-0.50)

    finish(fig, "data_distributions")
    print(f"[data] peptides n={len(pep)}  propeptides n={len(pro)}")
    print("[data] " + "  ".join(f"{g}:{v}" for g, v in gtop.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
