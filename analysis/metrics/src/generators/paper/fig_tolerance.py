#!/usr/bin/env python3
"""Appendix figure: F1 under a tightening boundary-match tolerance (paper style).

Ported from the Russian report generator
``analysis/metrics/src/generators/report_fig_tolerance.py``. The data logic --
which CSVs, the common-protein intersection over the pooled folds, and the
micro-averaged precision/recall/F1 at each tolerance -- is carried over
verbatim, so the numbers are identical; only labels, colours, fonts, sizing and
layout are new.

Input : analysis/metrics/clean_tol_true.csv   (one row per reference segment)
        analysis/metrics/clean_tol_pred.csv   (one row per predicted segment)
        Both carry per-segment match flags m0/m1/m2/m3 for tolerances +-0..+-3;
        rows are restricted to the proteins scored by *every* plotted model on
        the pooled folds {2, 5}.
Output: texs/ai4dd/figures/tolerance.png

    env/bin/python analysis/metrics/src/generators/paper/fig_tolerance.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from _common import LABEL, STYLE, apply, finish, panel, plt, textwidth  # noqa: E402

TRUE_CSV = "analysis/metrics/clean_tol_true.csv"
PRED_CSV = "analysis/metrics/clean_tol_pred.csv"

FOLDS = [2, 5]
TOLS = [3, 2, 1, 0]

# Plotted in the order the legend should read (worst -> best at +-3).
MODELS = [
    "baseline_esm2",
    "esmc_6b",
    "esmc6b_boundary",
    "esmc6b_3di_zeroctrl",
    "esmc6b_3di_gated_boundary",
]


# ------------------------------------------------------------------ data ---
def common(T, P, models, fold):
    """Proteins of `fold` that every model in `models` scored (true or pred)."""
    s = None
    for m in models:
        ts = set((f, p) for f, p in zip(T[T.model == m].fold, T[T.model == m].protein)
                 if f == fold)
        ps = set((f, p) for f, p in zip(P[P.model == m].fold, P[P.model == m].protein)
                 if f == fold)
        u = ts | ps
        s = u if s is None else s & u
    return s


def f1(T, P, cset, model, tol):
    """Micro-averaged segment F1 of `model` at tolerance `tol` over `cset`."""
    t = T[T.model == model]
    q = P[P.model == model]
    t = t[[(f, p) in cset for f, p in zip(t.fold, t.protein)]]
    q = q[[(f, p) in cset for f, p in zip(q.fold, q.protein)]]
    tp = t[f"m{tol}"].sum()
    fn = (t[f"m{tol}"] == 0).sum()
    fp = (q[f"m{tol}"] == 0).sum()
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


# ----------------------------------------------------------------- label ---
def spread(values, y0, y1, height_in, pad_pt=1.4, size_pt=7.0):
    """Vertical label positions >= one line apart, in data units.

    The two 3Di curves end 0.008 apart on a ~0.5 axis, so their end-of-line
    value labels overlapped in the old figure. Greedily push labels apart from
    the top down, keeping each within the axis, and return the nudged y for
    every input value (order preserved).
    """
    span = y1 - y0
    gap = (size_pt + pad_pt) / (height_in * 72.0) * span  # one text line, in data units
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = [0.0] * len(values)
    prev = None
    for i in order:
        y = values[i]
        if prev is not None and prev - y < gap:
            y = prev - gap
        out[i] = y
        prev = y
    # If the stack ran off the bottom, slide it all back up.
    low = min(out)
    if low < y0:
        out = [y + (y0 - low) for y in out]
    return out


# ------------------------------------------------------------------ main ---
def main() -> int:
    T = pd.read_csv(TRUE_CSV)
    P = pd.read_csv(PRED_CSV)

    cset = set()
    for fold in FOLDS:
        cset |= common(T, P, MODELS, fold)

    curves = {m: [f1(T, P, cset, m, t) for t in TOLS] for m in MODELS}
    retained = {m: [y / ys[0] for y in ys] for m, ys in curves.items()}

    apply()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(textwidth(0.95), 3.15), layout="constrained")

    x = list(range(len(TOLS)))
    for m in MODELS:
        colour, marker, dash = STYLE[m]
        kw = dict(color=colour, marker=marker, linestyle=dash, markersize=3.2,
                  linewidth=1.3, markeredgecolor="white", markeredgewidth=0.5)
        ax1.plot(x, curves[m], label=LABEL[m], **kw)
        ax2.plot(x, retained[m], **kw)

    ends = [retained[m][-1] for m in MODELS]
    lo, hi = min(ends) - 0.035, 1.02
    ax2.set_ylim(lo, hi)
    ax2.set_xlim(-0.15, 3.80)          # room for the labels past the last point

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([rf"$\pm{t}$" for t in TOLS])
        ax.set_xlabel("boundary-match tolerance")
        ax.grid(axis="x", alpha=0)
        ax.tick_params(length=0)

    ax1.set_ylabel("F1")
    ax2.set_ylabel("retained F1 share")
    # Two-line titles: at 0.475 linewidth a panel is ~2.2 in wide, and the
    # one-line form of (b) runs under the bold panel letter.
    panel(ax1, "a", "F1 as the tolerance\nis tightened")
    panel(ax2, "b", "Retained F1 share\n" + r"(F1@$\pm0$ / F1@$\pm3$)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               handlelength=2.4, columnspacing=1.6)

    # ---- panel (b): end-of-line value labels, de-overlapped -----------------
    # After every other artist: constrained layout only knows the final axes
    # height once the labels, titles and legend are in, and that height is what
    # sets the minimum vertical gap between two value labels.
    fig.canvas.draw()
    h_in = ax2.get_window_extent().height / fig.dpi
    ys = spread(ends, lo + 0.01, hi, h_in)
    for m, y_end, y_lab in zip(MODELS, ends, ys):
        colour = STYLE[m][0]
        ax2.annotate(
            f"{y_end:.2f}", xy=(3.03, y_end), xytext=(3.13, y_lab),
            color=colour, fontsize=7, fontweight="bold",
            va="center", ha="left", annotation_clip=False,
            arrowprops=dict(arrowstyle="-", linewidth=0.5, color=colour,
                            shrinkA=0, shrinkB=1),
        )

    finish(fig, "tolerance")

    for m in MODELS:
        ys_m = curves[m]
        print(f"  {LABEL[m]:46s} F1@+-3={ys_m[0]:.3f}  +-0={ys_m[-1]:.3f}  "
              f"retention={ys_m[-1] / ys_m[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
