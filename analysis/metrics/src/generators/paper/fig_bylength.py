#!/usr/bin/env python3
"""Appendix figure: F1 / precision / recall as a function of segment length.

Ported from the Russian report generator
`analysis/metrics/src/generators/bylength_fig.py` -- the data logic (inputs,
common-protein restriction, length binning, the >=25-true-segments cut, the
micro-averaged F1/P/R) is copied verbatim, so the numbers are identical. Only
the language, palette, fonts, sizing and layout are new.

Reads
    analysis/metrics/clean_tol_true.csv   per-true-segment matches (m3 = matched at +-3)
    analysis/metrics/clean_tol_pred.csv   per-predicted-segment matches
    analysis/metrics/adapter256_seg_true.csv \\ the gated(256) rung, produced
    analysis/metrics/adapter256_seg_pred.csv /  separately but in the same schema
    Single-split protocol, pooled held-out folds {2, 5}, corrected +-3 metric.

Writes
    texs/ai4dd/figures/bylength.png   (drawn at textwidth(0.95) = the fraction
    main.tex scales it to, so it lands in the PDF at scale 1)

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_bylength.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _common import LABEL, STYLE, apply, finish, panel, plt, textwidth

# ------------------------------------------------------------------ data ---
# Verbatim from bylength_fig.py.
CSV = "analysis/metrics/"
FOLDS = [2, 5]

T = pd.read_csv(CSV + "clean_tol_true.csv")
P = pd.read_csv(CSV + "clean_tol_pred.csv")
AT = pd.read_csv(CSV + "adapter256_seg_true.csv")
AP = pd.read_csv(CSV + "adapter256_seg_pred.csv")

# unified per-segment true/pred tables (cols: model,fold,protein,task,length/plen,m3)
Tt = pd.concat([T[["model", "fold", "protein", "task", "length", "m3"]],
                AT[["model", "fold", "protein", "task", "length", "m3"]]], ignore_index=True)
Pp = pd.concat([P[["model", "fold", "protein", "task", "plen", "m3"]],
                AP[["model", "fold", "protein", "task", "plen", "m3"]]], ignore_index=True)

MODELS = ["baseline_esm2", "esmc_6b", "esmc6b_boundary",
          "adapter256", "esmc6b_3di_gated_boundary"]


def pset(m):
    sub = Tt[(Tt.model == m) & (Tt.fold.isin(FOLDS))]
    return set(zip(sub.fold, sub.protein))


common = set.intersection(*[pset(m) for m in MODELS])

LB = ["5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-50"]


def lbin(L):
    e = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]
    for i in range(len(e) - 1):
        if e[i] <= L < e[i + 1]:
            return f"{e[i]}-{e[i + 1] - 1}"
    return None


def f1pr(tp, fn, fp):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r


def bylen(m):
    t = Tt[(Tt.model == m) & (Tt.fold.isin(FOLDS))].copy()
    q = Pp[(Pp.model == m) & (Pp.fold.isin(FOLDS))].copy()
    t = t[[(f, p) in common for f, p in zip(t.fold, t.protein)]]
    q = q[[(f, p) in common for f, p in zip(q.fold, q.protein)]]
    t["lb"] = t.length.map(lbin)
    q["lb"] = q.plen.map(lbin)
    res = {}
    for k in LB:
        tk = t[t.lb == k]
        qk = q[q.lb == k]
        if len(tk) < 25:
            continue
        res[k] = f1pr(tk.m3.sum(), (tk.m3 == 0).sum(), (qk.m3 == 0).sum())
    return res


CURVES = {m: bylen(m) for m in MODELS}
keys_present = [k for k in LB if k in CURVES[MODELS[0]]]
assert all([k for k in LB if k in CURVES[m]] == keys_present for m in MODELS), \
    "models disagree on which length bins survive the n>=25 cut"

# ------------------------------------------------------------------ plot ---
apply()
fig, axs = plt.subplots(1, 3, figsize=(textwidth(0.95), 3.35),
                        sharey=True, layout="constrained")

x = np.arange(len(keys_present))
PANELS = [("a", "F1", 0), ("b", "Precision", 1), ("c", "Recall", 2)]

for ax, (letter, title, mi) in zip(axs, PANELS):
    for m in MODELS:
        colour, marker, dash = STYLE[m]
        ax.plot(x, [CURVES[m][k][mi] for k in keys_present], label=LABEL[m],
                color=colour, marker=marker, linestyle=dash,
                markersize=3.0, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(keys_present, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", alpha=0)          # keep the grid off the categorical axis
    # panels (b)/(c) have no y tick labels (sharey), so the bold letter
    # needs a much smaller outdent or it drifts into the inter-panel gap.
    panel(ax, letter, title, dx=-0.085 if mi == 0 else -0.035)

axs[0].set_ylabel("metric value")
# One xlabel under the middle panel: it reads as centred under the row and,
# unlike fig.supxlabel, cannot collide with the outside legend below.
axs[1].set_xlabel("segment length (residues)")

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2,
           handlelength=2.4, columnspacing=1.6)

finish(fig, "bylength")

# ------------------------------------------------------------ printed check ---
print(f"[data] common proteins (fold, id) pairs: n={len(common)}; "
      f"bins={keys_present}")
for name, mi in (("F1", 0), ("P", 1), ("R", 2)):
    print(f"  -- {name} --")
    for m in MODELS:
        row = "  ".join(f"{k}:{CURVES[m][k][mi]:.3f}" for k in keys_present)
        print(f"    {LABEL[m]:44s} {row}")
