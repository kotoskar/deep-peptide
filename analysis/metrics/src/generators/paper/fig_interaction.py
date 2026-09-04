#!/usr/bin/env python3
"""Appendix figure: boundary head x embedding interaction (English, paper style).

Reads : ``analysis/metrics/interaction_perprotein_2026.csv`` -- per-protein
        corrected +-3 tp/fn/fp for the four single-split runs
        (``baseline_esm2``, ``esm2_boundary``, ``esmc_6b``, ``esmc6b_boundary``),
        written by ``analysis/metrics/src/generators/gpu_fig_data.py``.
Writes: ``texs/ai4dd/figures/interaction.png``. ``main.tex`` includes it with
        ``\\MaybeImage{figures/interaction.png}{0.9}``, so the figure is drawn
        exactly ``textwidth(0.9)`` wide and lands in the PDF at scale 1 with its
        8 pt type still 8 pt on paper.

Ported from ``analysis/metrics/src/generators/report_fig_interaction.py`` (the
Russian report figure). Its data logic is copied verbatim and, importantly, the
bootstrap draws are consumed in the same order from the same seeded generator,
so the numbers are identical: common ``(fold, protein)`` set across the four
models, micro-averaged F1 over pooled counts, 3000-sample bootstrap over
proteins (``default_rng(42)``) for the marginal 95% CI drawn on every point, and
the paired delta CI kept in the stdout printout because the body text cites it.
Only labels, colours, fonts, sizing and layout differ -- plus the grey note
under the axis, which says what the whiskers are (the dropped suptitle used to
carry "(95% CI)", and the LaTeX caption does not repeat it).

Usage (from the repo root):

    env/bin/python analysis/metrics/src/generators/paper/fig_interaction.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402

from _common import INK, STYLE, apply, finish, panel, textwidth  # noqa: E402

CSV = "analysis/metrics/interaction_perprotein_2026.csv"

# The four single-split runs the figure contrasts. The common-protein
# intersection is taken over all four so both slopes are measured on one set.
MS = ["baseline_esm2", "esm2_boundary", "esmc_6b", "esmc6b_boundary"]

# (legend label, plain run, +boundary run, STYLE key). Order matters: it fixes
# the order in which the shared RNG is consumed, hence the exact CI endpoints.
SERIES = [
    ("ESM-2 (1280)",    "baseline_esm2", "esm2_boundary",   "baseline_esm2"),
    ("ESM-C 6B (2560)", "esmc_6b",       "esmc6b_boundary", "esmc_6b"),
]

rng = np.random.default_rng(42)


# ------------------------------------------------------------------ data ---
D = pd.read_csv(CSV)


def _key(m):
    return set(zip(D[D.model == m].fold, D[D.model == m].protein))


common = set.intersection(*[_key(m) for m in MS])


def arr(m):
    s = D[D.model == m].copy()
    s = s[[(f, p) in common for f, p in zip(s.fold, s.protein)]]
    s = s.sort_values(["fold", "protein"])
    return s[["tp", "fn", "fp"]].to_numpy(float)


def f1(a):
    tp, fn, fp = a.sum(0)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


A = {m: arr(m) for m in MS}


def marg(m, B=3000):
    """Marginal 95% bootstrap CI on one point (both ends of a line get a bar)."""
    a = A[m]
    n = len(a)
    pt = f1(a)
    bs = np.array([f1(a[s]) for s in (rng.integers(0, n, n) for _ in range(B))])
    return pt, *np.percentile(bs, [2.5, 97.5])


def paired_delta(head, plain, B=3000):
    """Paired delta CI -- not drawn, but printed because the body text cites it."""
    H, P = A[head], A[plain]
    n = len(H)
    pt = f1(H) - f1(P)
    bs = np.array([f1(H[s]) - f1(P[s]) for s in (rng.integers(0, n, n) for _ in range(B))])
    return pt, *np.percentile(bs, [2.5, 97.5])


# ------------------------------------------------------------------ plot ---
apply()
fig, ax = plt.subplots(figsize=(textwidth(0.9), 2.55), layout="constrained")
# Room under the axis for the error-bar note; the figure grew by the same
# amount, so the plot area is unchanged. Width stays textwidth(0.9).
fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))  # (left, bottom, w, h)
x = [0, 1]

for label, plain, head, style_key in SERIES:
    colour, marker, dash = STYLE[style_key]
    p0, l0, h0 = marg(plain)
    p1, l1, h1 = marg(head)
    dt = p1 - p0
    ax.errorbar(x, [p0, p1], yerr=[[p0 - l0, p1 - l1], [h0 - p0, h1 - p1]],
                color=colour, marker=marker, linestyle=dash, label=label,
                linewidth=1.4, markersize=4.0, markeredgecolor="white",
                markeredgewidth=0.6, capsize=2.2, elinewidth=0.8, capthick=0.8,
                zorder=3)
    ax.annotate(f"Δ = {dt:+.3f}", (1, p1), textcoords="offset points",
                xytext=(7, 0), ha="left", va="center", color=colour, zorder=4)
    pd_, plo, phi = paired_delta(head, plain)
    print(f"{label}: plain={p0:.3f} [{l0:.3f},{h0:.3f}]  head={p1:.3f} "
          f"[{l1:.3f},{h1:.3f}]  d={dt:+.3f}  (paired CI [{plo:+.3f},{phi:+.3f}])")

ax.set_xticks(x)
ax.set_xticklabels(["without boundary head", "+ boundary head"])
# Room on the right for the delta labels; the old figure nearly clipped them.
ax.set_xlim(-0.32, 1.62)
ax.set_ylim(0.53, 0.70)
ax.set_ylabel("F1")
ax.grid(axis="x", alpha=0)          # categorical axis, as in the Russian original
ax.tick_params(length=0)
leg = ax.legend(loc="upper left", title="embedding", handlelength=2.4,
                title_fontsize=plt.rcParams["legend.fontsize"], borderaxespad=0.2)
leg.get_title().set_ha("left")
panel(ax, None, "Boundary head × embedding")

# What the whiskers are. Figure 1 draws the fold-to-fold standard deviation,
# these are a percentile bootstrap over proteins, so saying it matters.
fig.text(0.012, 0.012,
         "error bars \u2014 95% percentile bootstrap CI, resampling proteins (not folds)",
         fontsize=6.5, style="italic", color=INK, ha="left", va="bottom")

finish(fig, "interaction")
print(f"interaction done (n_common={len(common)})")
