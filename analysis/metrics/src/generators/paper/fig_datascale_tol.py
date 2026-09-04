#!/usr/bin/env python3
"""Appendix figure: boundary sharpness as a function of training-set size.

Ported from the Russian report generator
`analysis/metrics/src/generators/datascale_tol_plot.py`. The data logic (which
CSV, which grouping, the pooled-protein bootstrap, the aggregation into F1 and
into the retained-F1 ratio) is carried over verbatim, so every number is the
one the report already reports; only labels, colours, fonts, sizing and layout
are new. English + AIRI palette + true print size, per `_common.py`.

Input : analysis/metrics/datascale_tol_perprotein.csv
        per-protein tp/fn/fp at tolerance +-3 and +-0, for each model tag
        ("baseline", "proj", "3di"), each training-set fraction (40..100 %)
        and each of the two pooled test folds ({2} and {5}).
Output: texs/ai4dd/figures/datascale_tolerance.png
        (a) the CSV tag "proj" -- runs/scale_proj_* and, at 100 %,
            runs/2026_esmc6b_3di_zeroctrl, i.e. LABEL["esmc6b_3di_zeroctrl"],
            NOT the plain ESM-C 6B base of Figure 1 (runs/2026_esmc_6b) --
            scored at tolerance +-3 and at +-0 vs. the number of training
            proteins, with 95 % bootstrap CI bands;
        (b) the retained F1 share F1@+-0 / F1@+-3 for three models, same x.

Run from the repo root:
    env/bin/python analysis/metrics/src/generators/paper/fig_datascale_tol.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from _common import (AIRI, LABEL, MUTE, STYLE, apply, finish,  # noqa: E402
                     panel, textwidth)
import matplotlib.pyplot as plt                        # noqa: E402

CSV = "analysis/metrics/datascale_tol_perprotein.csv"
NBOOT = 2000

# Panel (a): the two tolerances of ONE model. Deliberately not model colours --
# these two curves are the same model scored two ways.
TOL_STYLE = {
    3: (AIRI["A"],   "o", "-",           "tolerance \u00b13"),
    0: (AIRI["E21"], "s", (0, (4, 1.5)), "tolerance \u00b10"),
}
# The CSV tag whose two tolerances panel (a) draws. Proven from
# ../datascale_tol2.py (the script that wrote the CSV): "proj" is
# runs/scale_proj_{40,55,70,85} plus runs/2026_esmc6b_3di_zeroctrl at 100 %,
# all of them model=lstmcnncrf_gated3di_boundary with seq_proj_size=256 on
# embeddings_esmc6b_3dizero. That is LABEL["esmc6b_3di_zeroctrl"], a different
# model from the plain ESM-C 6B base of Figure 1 (runs/2026_esmc_6b:
# model=lstmcnncrf, embeddings_esmc6b, no boundary head, no adapter).
PANEL_A_TAG = "proj"
PANEL_A_KEY = "esmc6b_3di_zeroctrl"

# Panel (b): CSV tag -> (_common model key, legend label).
MODELS = [
    ("baseline", "baseline_esm2",             LABEL["baseline_esm2"]),
    ("proj",     PANEL_A_KEY,                 LABEL[PANEL_A_KEY]),
    ("3di",      "esmc6b_3di_gated_boundary", LABEL["esmc6b_3di_gated_boundary"]),
]
# Vertical stagger (points) for the right-hand delta labels: "proj" and "3di"
# end within 0.01 of each other and would otherwise print on top of each other.
DY = {"baseline": 0.0, "proj": 4.5, "3di": -4.5}


# --------------------------------------------------------------- data logic ---
# Verbatim from datascale_tol_plot.py.

def f1(tp, fn, fp):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def boot(sub, rng, B=NBOOT):
    a3 = sub[["tp3", "fn3", "fp3"]].to_numpy(float)
    a0 = sub[["tp0", "fn0", "fp0"]].to_numpy(float)
    n = len(sub)
    pt3 = f1(*a3.sum(0))
    pt0 = f1(*a0.sum(0))
    ptr = pt0 / pt3 if pt3 else 0
    b3 = np.empty(B)
    b0 = np.empty(B)
    br = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        s3 = a3[s].sum(0)
        s0 = a0[s].sum(0)
        x3 = f1(*s3)
        x0 = f1(*s0)
        b3[i] = x3
        b0[i] = x0
        br[i] = x0 / x3 if x3 else 0
    q = lambda v: np.percentile(v, [2.5, 97.5])         # noqa: E731
    return (pt3, *q(b3)), (pt0, *q(b0)), (ptr, *q(br))


def load():
    D = pd.read_csv(CSV)
    rng = np.random.default_rng(42)
    rec = {}
    for (tag, frac), sub in D.groupby(["tag", "frac"]):
        ntr = int(sub.n_train.iloc[0])
        rec[(tag, frac)] = (ntr, *boot(sub, rng))
    return rec


def series(rec, tag):
    fr = sorted(f for (t, f) in rec if t == tag)
    return [rec[(tag, f)] for f in fr]


# ------------------------------------------------------------------ drawing ---
# Greek and sign glyphs are literal characters, never mathtext: mathtext is
# typeset in DejaVu Sans and would clash with the serif body of the figure.
# (Figure 1's mathtext tolerance ticks are a separate, deliberate case.)

def delta_label(d):
    """`\u0394 = +0.05`, with a real U+2212 minus for negative deltas."""
    return f"\u0394 = {d:+.2f}".replace("-", "\u2212")


def curve(ax, xs, pts, los, his, colour, marker, dash, label):
    ax.plot(xs, pts, color=colour, marker=marker, linestyle=dash, label=label,
            linewidth=1.2, markersize=3.2, markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)
    ax.fill_between(xs, los, his, color=colour, alpha=0.15, linewidth=0, zorder=1)


def main() -> int:
    rec = load()
    apply()
    # One column, two rows: at 5.2 in total width a side-by-side pair leaves
    # ~2.4 in per panel, which neither panel title nor panel (b)'s y-label fits
    # into at 8 pt. Height is free (appendix figure), width is not.
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(textwidth(0.95), 5.1), layout="constrained")

    # ---- (a) one model, two tolerances -----------------------------------
    s = series(rec, PANEL_A_TAG)
    xs = [r[0] for r in s]
    for idx, tol in ((1, 3), (2, 0)):
        colour, marker, dash, label = TOL_STYLE[tol]
        pt = [r[idx][0] for r in s]
        lo = [r[idx][1] for r in s]
        hi = [r[idx][2] for r in s]
        curve(ax1, xs, pt, lo, hi, colour, marker, dash, label)
        d = pt[-1] - pt[0]
        ax1.annotate(delta_label(d), (xs[-1], pt[-1]),
                     textcoords="offset points", xytext=(5, 0), color=colour,
                     fontsize=7, ha="left", va="center", zorder=5)
        print(f"  (a) tol {tol:+d}: " + "  ".join(f"{v:.3f}" for v in pt)
              + f"   delta={d:+.3f}")
    ax1.set_xlim(1900, xs[-1] + 900)
    ax1.set_ylabel("F1")
    ax1.legend(loc="center right", handlelength=2.2)
    # The full "F1 at tolerances +-3 and +-0" runs to 103 % of the axes width
    # once the configuration name is in front of it; the legend right below
    # spells out "tolerance +-3" / "tolerance +-0", so the title keeps the
    # short form and fits at 89 %.
    panel(ax1, "a", f"{LABEL[PANEL_A_KEY]}: F1 at \u00b13 and \u00b10")

    # ---- (b) retained F1 share, three models -----------------------------
    last = 0
    for tag, key, label in MODELS:
        colour, marker, dash = STYLE[key]
        s = series(rec, tag)
        xs = [r[0] for r in s]
        last = max(last, xs[-1])
        pt = [r[3][0] for r in s]
        lo = [r[3][1] for r in s]
        hi = [r[3][2] for r in s]
        curve(ax2, xs, pt, lo, hi, colour, marker, dash, label)
        d = pt[-1] - pt[0]
        ax2.annotate(delta_label(d), (xs[-1], pt[-1]),
                     textcoords="offset points", xytext=(5, DY[tag]),
                     color=colour, fontsize=7, ha="left", va="center", zorder=5)
        print(f"  (b) {label:44s} " + "  ".join(f"{v:.3f}" for v in pt)
              + f"   delta={d:+.3f}")
    ax2.set_xlim(1900, last + 900)
    # Two lines: on one line this label is as tall as the axes and runs into
    # the bold panel letter.
    ax2.set_ylabel("retained F1 share\n(F1@\u00b10 / F1@\u00b13)")
    panel(ax2, "b", "Retained F1 share at an exact match")
    # The suptitle that carried this is gone and the caption in main.tex says
    # neither, so the two facts live inside the figure now. Panel (b)'s
    # lower-right corner is the only region no band reaches.
    # y is chosen so the note sits between the 0.40 and 0.45 grid lines rather
    # than on one of them; nothing is plotted in that corner.
    ax2.text(0.985, 0.10,
             "bands: 95 % bootstrap CI (both panels);"
             "   test set: pooled folds {2} and {5}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7, color=MUTE, zorder=5)
    handles, labels = ax2.get_legend_handles_labels()

    for ax in (ax1, ax2):
        ax.set_xlabel("number of proteins in the training set")

    # Panel (b)'s three configuration names are far too long for a 2.5 in wide
    # panel, so they go under the figure; panel (a)'s two-entry legend stays
    # inside its own panel, where it cannot be mistaken for this one.
    fig.legend(handles, labels, loc="outside lower center", ncols=1,
               handlelength=2.4, labelspacing=0.35, borderaxespad=0.2)

    finish(fig, "datascale_tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
