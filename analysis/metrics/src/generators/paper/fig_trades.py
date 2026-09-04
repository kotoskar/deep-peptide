#!/usr/bin/env python3
"""Appendix figure: what the 3Di channel and the bond loss trade away.

Two panels of grouped bars, each comparing one addition against its own proper
control -- 3Di against the zeroed-3Di model, the bond loss against the same
boundary model without it -- as the change in segment precision and recall,
split by segment type, with 95 % bootstrap CIs.

Input : analysis/metrics/clean_tol_true.csv   (annotated segments, +-3 match flags)
        analysis/metrics/clean_tol_pred.csv   (predicted segments, +-3 match flags)
Output: texs/ai4dd/figures/trades.png

Ported from the Russian report generator
`analysis/metrics/src/generators/trades_fig.py` (the two-panel version). The
data logic -- pooled folds {2, 5}, per-protein tp/fn/fp pooling on the +-3
match column, paired protein-level bootstrap with B=3000 draws off a single
`default_rng(42)` stream -- is reproduced verbatim, including the order in
which the bootstrap consumes that stream, so the numbers are unchanged. Only
the labels, colours, fonts, sizing and layout differ.

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_trades.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    INK, MUTE, PRECISION_COLOR, RECALL_COLOR, apply, finish, panel, plt, textwidth,
)

T = pd.read_csv("analysis/metrics/clean_tol_true.csv")
P = pd.read_csv("analysis/metrics/clean_tol_pred.csv")

rng = np.random.default_rng(42)
FOLDS = [2, 5]
B = 3000

TASKS = ("pep", "propep")
TASK_LABEL = {"pep": "peptides", "propep": "propeptides"}
# (key, colour, legend label) -- Greek Delta, not the ASCII lookalike.
SERIES = (("p", PRECISION_COLOR, "Δ precision"),
          ("r", RECALL_COLOR, "Δ recall"))


# ------------------------------------------------------------------- data ---

def metric(arr, kind):
    tp, fn, fp = arr.sum(0)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0}[kind]


def ppm(m, task):
    """Per-protein tp/fn/fp for model `m` on `task`, pooled over FOLDS."""
    t = T[(T.model == m) & (T.fold.isin(FOLDS)) & (T.task == task)]
    q = P[(P.model == m) & (P.fold.isin(FOLDS)) & (P.task == task)]
    tg = t.groupby(["fold", "protein"]).m3.agg(tp="sum", ntrue="size")
    pg = q.groupby(["fold", "protein"]).m3.agg(mp="sum", npred="size")
    d = tg.join(pg, how="outer").fillna(0.0)
    d["fn"] = d.ntrue - d.tp
    d["fp"] = d.npred - d.mp
    return d[["tp", "fn", "fp"]]


def paired(WIN, ZERO, kind, task):
    """Point estimate and 95 % CI of metric(WIN) - metric(ZERO), proteins paired."""
    A = ppm(WIN, task)
    Bn = ppm(ZERO, task)
    idx = A.index.intersection(Bn.index)
    A = A.loc[idx].to_numpy()
    Bm = Bn.loc[idx].to_numpy()
    n = len(idx)
    pt = metric(A, kind) - metric(Bm, kind)
    bs = np.array([metric(A[s], kind) - metric(Bm[s], kind)
                   for s in (rng.integers(0, n, n) for _ in range(B))])
    return pt, *np.percentile(bs, [2.5, 97.5])


# ---------------------------------------------------------------- drawing ---

def draw(ax, WIN, ZERO, letter, title, tag, ylab=None, dx=-0.085):
    x = np.arange(len(TASKS))
    w = 0.32
    for (kind, colour, name), off in zip(SERIES, (-w / 2, w / 2)):
        pts, los, his = [], [], []
        for task in TASKS:
            pt, lo, hi = paired(WIN, ZERO, kind, task)
            print(f"[{tag}] {TASK_LABEL[task]:>11s} d{kind}: "
                  f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            pts.append(pt)
            los.append(pt - lo)
            his.append(hi - pt)
        ax.bar(x + off, pts, w, color=colour, label=name, zorder=3,
               edgecolor="white", linewidth=0.6)
        # The precision bars are dark teal, so a plain ink whisker drawn on
        # top of them is invisible (contrast ~1.1:1) and the reader cannot see
        # where the interval ends. A white halo keeps the same ink colour
        # legible both inside a bar and against the white background outside.
        eb = ax.errorbar(x + off, pts, yerr=[los, his], fmt="none", ecolor=INK,
                         elinewidth=0.9, capsize=0, zorder=4)
        for coll in eb[2]:
            coll.set_path_effects([pe.withStroke(linewidth=1.9,
                                                 foreground="white")])

    ax.axhline(0, color=MUTE, linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_xlim(-0.62, len(TASKS) - 0.38)
    if ylab:
        ax.set_ylabel(ylab)
    ax.grid(axis="x", alpha=0)          # bar chart: no vertical grid
    ax.tick_params(length=0)
    panel(ax, letter, title, dx=dx)


def main() -> int:
    apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(textwidth(0.95), 2.95),
                                 sharey=True, layout="constrained")
    # Reserve a strip at the bottom for the legend and, under it, the note on
    # the error bars: constrained layout sees neither a `fig.text` nor a
    # manually anchored legend, so both are placed by hand inside that strip.
    # `rect` is (left, bottom, width, height), so the height shrinks with it.
    fig.get_layout_engine().set(rect=(0, 0.155, 1, 0.845))
    # The bootstrap draws from one rng stream, so panel order is load-bearing.
    draw(a1, "esmc6b_3di_gated_boundary", "esmc6b_3di_zeroctrl",
         "a", "effect of adding 3Di", "3Di", ylab="metric difference")
    draw(a2, "esmc6b_boundary_bond", "esmc6b_boundary",
         "b", "effect of adding the bond loss", "bond", dx=-0.03)

    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.068), ncol=2,
               handlelength=1.6, columnspacing=1.8)
    # The figure carries no suptitle any more, and the caption does not define
    # the whiskers, so the definition lives here. Literal Greek/typographic
    # glyphs, never mathtext: mathtext would render sans-serif.
    fig.text(0.5, 0.028,
             "error bars \u2014 95 % paired protein-level bootstrap CI "
             "(B = 3000), pooled held-out folds 2 and 5",
             ha="center", va="bottom", fontsize=6.5, color=MUTE)
    finish(fig, "trades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
