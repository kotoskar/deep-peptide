#!/usr/bin/env python3
"""Appendix figure: segment F1 as a function of the training-set size.

Input  : analysis/metrics/datascale_curve.csv -- columns tag,n_train,f1,lo,hi,
         written by ../datascale_curve.py: F1 pooled over held-out folds {2} and
         {5}, lo/hi the 2.5/97.5 percentiles of a bootstrap over proteins (95%
         CI). Read verbatim, exactly as the Russian generator does.
Output : texs/ai4dd/figures/datascale_curve.png

Ported from analysis/metrics/src/generators/datascale_plot.py (Russian report
version). The data logic -- which CSV, which rows, which error bars, which
end-to-end delta -- is unchanged; only the language, palette, fonts, sizing and
layout differ. The figure is drawn at textwidth(0.7), the fraction main.tex
scales it to (`\\MaybeImage{figures/datascale_curve.png}{0.7}`), so it lands in
the PDF at scale 1 and the 8 pt type stays 8 pt on paper.

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_datascale.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import LABEL, STYLE, apply, finish, panel, textwidth  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CSV = pathlib.Path("analysis/metrics/datascale_curve.csv")

# CSV tag -> (model key in LABEL/STYLE, delta-label vertical offset in points).
# The two ESM-C curves end 0.002 F1 apart, so their delta labels have to be
# pushed apart by hand or they land on top of each other (as in the old figure).
SERIES = [
    ("baseline", "baseline_esm2", 0.0),
    ("proj", "esmc6b_3di_zeroctrl", 7.0),
    ("3di", "esmc6b_3di_gated_boundary", -7.0),
]

# One wording per configuration across the whole paper: the zeroed-3Di control is
# "3Di zeroed" everywhere (_common.LABEL), never the local synonym "3Di off" the
# Russian original used here -- the two figures sit two pages apart.
LABEL_OVERRIDE = {}


def main() -> int:
    df = pd.read_csv(CSV)

    apply()
    fig, ax = plt.subplots(figsize=(textwidth(0.7), 3.15), layout="constrained")

    # All three curves end within 100 proteins of each other, so the delta
    # labels share one right-hand column, past the widest error bar.
    anchor_x = df.n_train.max()

    annotations = []
    for tag, key, dy in SERIES:
        r = df[df.tag == tag].sort_values("n_train")
        x = r.n_train.to_numpy()
        f1 = r.f1.to_numpy()
        colour, marker, dash = STYLE[key]
        ax.errorbar(
            x, f1,
            yerr=[f1 - r.lo.to_numpy(), r.hi.to_numpy() - f1],
            label=LABEL_OVERRIDE.get(key, LABEL[key]),
            color=colour, marker=marker, linestyle=dash,
            markersize=3.4, linewidth=1.3, markeredgecolor="white",
            markeredgewidth=0.4, capsize=2, elinewidth=0.7, zorder=3,
        )
        annotations.append(ax.annotate(
            f"Δ = {f1[-1] - f1[0]:+.2f}", (anchor_x, f1[-1]),
            textcoords="offset points", xytext=(7, dy),
            fontsize=7, color=colour, ha="left", va="center", zorder=4,
        ))

    ax.set_xlabel("number of proteins in the training set")
    ax.set_ylabel("F1")
    ax.set_ylim(0.44, 0.72)
    ax.set_xlim(1850, anchor_x + 250)
    # The Russian original keeps a faint vertical grid (x is continuous here,
    # not categorical), and the horizontal grid comes from paperstyle.
    ax.grid(axis="x", alpha=0.3)
    panel(ax, None, "F1 vs. training-set size")

    # Grow the right margin until every delta label sits inside the axes with a
    # 6 pt gap to the spine; the old figure let one of them run over the spine.
    pad_px = 6 * fig.dpi / 72
    for _ in range(8):
        fig.canvas.draw()
        right = ax.get_window_extent().x1
        over = max(a.get_window_extent().x1 - right for a in annotations) + pad_px
        if abs(over) < 1:
            break
        x0, x1 = ax.get_xlim()
        per_px = (x1 - x0) / max(ax.get_window_extent().width, 1.0)
        ax.set_xlim(x0, x1 + over * per_px)

    fig.legend(loc="outside lower center", ncols=1, handlelength=2.6,
               labelspacing=0.35, handletextpad=0.6)

    finish(fig, "datascale_curve")
    for tag, key, _ in SERIES:
        r = df[df.tag == tag].sort_values("n_train")
        pts = "  ".join(f"{n}:{v:.3f}" for n, v in zip(r.n_train, r.f1))
        print(f"  {LABEL_OVERRIDE.get(key, LABEL[key]):42s} {pts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
