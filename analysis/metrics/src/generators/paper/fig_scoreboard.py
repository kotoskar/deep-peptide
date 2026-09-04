#!/usr/bin/env python3
"""Appendix figure: single-split scoreboard -- F1, precision and recall at +-3.

Seven models on one set of rows, each drawn as a point estimate with its 95 %
bootstrap interval, sorted by F1. The two gated *controls* (3Di zeroed, gated
projector left at full width) are drawn hollow; the dashed rule marks the step
into the gated cluster, which the ladder does not isolate on its own -- the
isolated size of that step comes from the dedicated ablation and is stated in
the note under the panels.

Ported from the Russian report generator
``analysis/metrics/src/generators/report_figs_v2.py`` (the ``SCOREBOARD`` block
only). The data logic -- the common-protein intersection, the per-protein
tp/fn/fp pooling at tolerance 3, the 2 000-replicate bootstrap and the order in
which the bootstrap consumes the seeded RNG -- is a verbatim port, so the
numbers are identical to that figure. Only labels, colours, fonts, sizing and
layout differ: English text, AIRI palette, drawn at true print size, and the
figure-level title dropped per the house style of `_common.py`. The tolerance,
the bootstrap interval and n live in the LaTeX caption; the two things the
caption does *not* carry -- what a hollow marker means, and the caveat that the
gated-adapter gain rests mostly on fold 5 -- are kept in the note under the
panels.

Input : analysis/metrics/clean_tol_true.csv
        analysis/metrics/clean_tol_pred.csv
Output: texs/ai4dd/figures/scoreboard.png  (\\MaybeImage{...}{0.9} in main.tex)

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_scoreboard.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    INK, MUTE, LABEL, STYLE, apply, textwidth, panel, finish, plt,
)

FRAC = 0.9                       # \MaybeImage{figures/scoreboard.png}{0.9}
CSV = pathlib.Path("analysis/metrics")
FOLDS = [2, 5]
TOL_COL = "m3"                   # the corrected +-3 match column

# ------------------------------------------------------------- data logic ---
# Verbatim from report_figs_v2.py: same seed, same call order, same numbers.
rng = np.random.default_rng(42)
T = pd.read_csv(CSV / "clean_tol_true.csv")
P = pd.read_csv(CSV / "clean_tol_pred.csv")


def keyset(m):
    t = set((f, p) for f, p in zip(T[T.model == m].fold, T[T.model == m].protein)
            if f in FOLDS)
    pr = set((f, p) for f, p in zip(P[P.model == m].fold, P[P.model == m].protein)
             if f in FOLDS)
    return t | pr


def common(models):
    s = None
    for m in models:
        s = keyset(m) if s is None else s & keyset(m)
    return s


def f1pr(tp, fn, fp):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r


# (model key, role). "control" rows get a hollow marker.
LAD_M = [("baseline_esm2", "ladder"), ("esmc_600m", "ladder"), ("esmc_6b", "ladder"),
         ("esmc6b_boundary", "ladder"), ("esmc6b_3di_gated_boundary", "ladder"),
         ("esmc6b_3di_zeroctrl", "control"), ("esmc6b_3di_nocompress", "control")]
cprot = common([m for m, _ in LAD_M])


def model_pp(m, which):
    """Per-(fold, protein) tp/fn/fp at tolerance 3 on the common protein set."""
    t = T[(T.model == m) & (T.fold.isin(FOLDS))]
    q = P[(P.model == m) & (P.fold.isin(FOLDS))]
    if which != "all":
        t = t[t.task == which]
        q = q[q.task == which]
    t = t[[(f, p) in cprot for f, p in zip(t.fold, t.protein)]]
    q = q[[(f, p) in cprot for f, p in zip(q.fold, q.protein)]]
    tg = t.groupby(["fold", "protein"])[TOL_COL].agg(tp="sum", ntrue="size")
    pg = q.groupby(["fold", "protein"])[TOL_COL].agg(mp="sum", npred="size")
    d = tg.join(pg, how="outer").fillna(0.0)
    d["fn"] = d.ntrue - d.tp
    d["fp"] = d.npred - d.mp
    return d[["tp", "fn", "fp"]].to_numpy()


def metric(arr, kind):
    tp, fn, fp = arr.sum(0)
    f, p, r = f1pr(tp, fn, fp)
    return {"f1": f, "p": p, "r": r}[kind]


def marg(m, kind, B=2000):
    """Point estimate and 95 % percentile bootstrap interval over proteins."""
    arr = model_pp(m, "all")
    n = len(arr)
    pt = metric(arr, kind)
    bs = np.array([metric(arr[rng.integers(0, n, n)], kind) for _ in range(B)])
    return (pt, *np.percentile(bs, [2.5, 97.5]))


# ----------------------------------------------------------------- drawing ---
PANELS = [("f1", "a", "F1"), ("p", "b", "Precision"), ("r", "c", "Recall")]
XLIM = (0.51, 0.82)              # shared, so the three panels stay comparable
XTICKS = [0.55, 0.65, 0.75]


def rowlabel(key):
    """LABEL[key], wrapped onto two lines so the label column stays narrow."""
    return LABEL[key].replace(" + gated", "\n+ gated")


def main():
    apply()

    keys = [m for m, _ in LAD_M]
    # Same RNG consumption order as the Russian original: the sort key is
    # evaluated once per model first, then each panel re-evaluates all seven.
    f1order = sorted(range(len(LAD_M)), key=lambda i: marg(keys[i], "f1")[0])
    ypos = {i: yi for yi, i in enumerate(f1order)}
    oi = keys.index("esmc6b_boundary")          # last rung below the gated cluster

    fig, axs = plt.subplots(1, 3, figsize=(textwidth(FRAC), 3.60),
                            sharey=True, layout="constrained")
    # (left, bottom, width, height): keep the bottom strip free for the
    # three-line note. The figure grew taller to pay for it, so the panels
    # keep their height; the width must stay textwidth(FRAC).
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))

    for ax, (kind, letter, title) in zip(axs, PANELS):
        vals = [marg(m, kind) for m, _ in LAD_M]
        for i, yi in ypos.items():
            key, role = LAD_M[i]
            col = STYLE[key][0]
            pt, lo, hi = vals[i]
            # Full opacity: an alpha-blended interval in a light hue (the tan
            # of the gated(2560) control) washes out to the grid colour and
            # the reader loses the interval entirely. The thinner line keeps
            # the point estimate dominant instead.
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.7,
                    solid_capstyle="round", zorder=2)
            # One marker shape for every row, as in the original: here the
            # filled/hollow contrast is the only thing the shape has to carry,
            # since the row label already names the model.
            if role == "control":
                ax.plot(pt, yi, "o", mfc="white", mec=col, mew=1.4, ms=5.0,
                        zorder=3)
            else:
                ax.plot(pt, yi, "o", color=col, ms=5.0, mec="white", mew=0.7,
                        zorder=3)
            ax.text(hi + 0.006, yi, f"{pt:.3f}", va="center", ha="left",
                    fontsize=6.5, color=MUTE, zorder=4)

        # The step into the gated cluster is not isolated by this ladder.
        ax.axhline(ypos[oi] + 0.5, color=MUTE, ls="--", lw=0.8, alpha=0.7,
                   zorder=1)

        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        ax.set_ylim(-0.65, len(LAD_M) - 0.35)
        ax.set_yticks(range(len(LAD_M)))
        ax.grid(axis="y", alpha=0)
        ax.tick_params(length=0, pad=2)
        panel(ax, letter, title)

    axs[0].set_yticklabels([rowlabel(LAD_M[i][0]) for i in f1order],
                           fontsize=7, linespacing=1.15)

    # What the LaTeX caption does not say. Three lines because 6.5 pt italic
    # runs out of room at ~100 characters across textwidth(0.9).
    fig.text(0.012, 0.012,
             "hollow markers \u2014 gated-model controls, not rungs of the ladder\n"
             "dashed rule \u2014 gated adapter: +0.022 (isolated by ablation),\n"
             "    significant on average, but the gain is mostly on fold 5",
             fontsize=6.5, style="italic", color=INK, ha="left", va="bottom",
             linespacing=1.3)

    finish(fig, "scoreboard")
    print(f"[fig] n={len(cprot)} proteins, folds {FOLDS}, tolerance +-3")


if __name__ == "__main__":
    main()
