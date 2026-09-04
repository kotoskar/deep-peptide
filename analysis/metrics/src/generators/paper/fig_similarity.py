#!/usr/bin/env python3
"""Appendix figure: what drives recall -- genus representation vs. similarity.

Ported from the Russian report generator
``analysis/metrics/src/generators/sim_fig_report.py``.  The *data logic* (which
CSVs, which protein intersection, which genus-abundance counting, which bins,
the seeded 2000-draw bootstrap and the rank-correlation) is copied verbatim, so
the numbers are identical; only labels, colours, fonts, sizing and layout are
new.  The figure is drawn at its true print width -- ``textwidth(0.95)``, the
fraction ``\\MaybeImage{figures/similarity.png}{0.95}`` uses in ``main.tex`` --
so 8 pt in the figure is 8 pt on paper.

Reads
-----
* ``analysis/similarity/seg_matched_2026.csv``      -- per-segment hit/miss, 2026 models
* ``analysis/similarity/seg_matched_zeroctrl.csv``  -- same, for the 3Di-zeroed control
* ``analysis/similarity/identity_2026.csv``         -- max. identity of a segment to train
* ``data/uniprot_2026/labeled_sequences.csv``       -- organism + segment coordinates
* ``data/uniprot_2026/graphpart_assignments.csv``   -- homology-partition fold per protein

Writes
------
``texs/ai4dd/figures/similarity.png``

Panels: (a) recall against how many training segments the genus contributes
(flat -- representation does not explain recall), (b) recall against the maximum
identity of the segment itself to any training segment (rises steeply for every
model).  Shaded bands are 95 % bootstrap CIs; grey ``n=`` annotations give the
per-bin segment count.

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_similarity.py
"""
from __future__ import annotations

import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _common import (LABEL, MUTE, STYLE, apply, finish, panel,
                     plt, textwidth)

# ------------------------------------------------------------------ data ---
# Everything below the "data logic" banner is the report generator's logic,
# unchanged; see the module docstring.
rng = np.random.default_rng(42)

MODELS = ["baseline_esm2", "esmc6b_3di_zeroctrl", "esmc6b_3di_gated_boundary"]

# Panel letter outdent: these panels are full figure width, so the default
# -0.085 axes fractions lands on top of the rotated y-axis label.
DX = -0.052

ABINS = [0, 5, 20, 100, 1e9]
ALBL = ["1–5", "6–20", "21–100", "100+"]
IBINS = [0, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0001]
ILBL = ["<0.30", "0.30–0.40", "0.40–0.50",
        "0.50–0.60", "0.60–0.70", "\u22650.70"]


def _parse(s):
    return re.findall(r"\((\d+)-(\d+)\)", s) if isinstance(s, str) else []


def load():
    """Segment-level table with genus train-abundance and identity bins."""
    sm = pd.concat([pd.read_csv("analysis/similarity/seg_matched_2026.csv"),
                    pd.read_csv("analysis/similarity/seg_matched_zeroctrl.csv")],
                   ignore_index=True)
    idn = pd.read_csv("analysis/similarity/identity_2026.csv")
    d = sm.merge(idn, on=["task", "seq"], how="left")
    d = d[d.model.isin(MODELS)]
    # Compare the three models on exactly the same proteins.
    common = set.intersection(*[set(d[d.model == m].protein) for m in MODELS])
    d = d[d.protein.isin(common)]

    # Genus train-abundance: segments (both types) the genus contributes to the
    # training folds 0, 3, 4, 6.
    trf = pd.read_csv("data/uniprot_2026/labeled_sequences.csv").merge(
        pd.read_csv("data/uniprot_2026/graphpart_assignments.csv")
          .rename(columns={"AC": "protein_id", "cluster": "fold"}),
        on="protein_id")
    trf["genus"] = trf.organism.fillna("?").str.split().str[0]
    tcnt: dict[str, int] = {}
    for _, r in trf[trf.fold.isin([0, 3, 4, 6])].iterrows():
        n = len(_parse(r["coordinates"])) + len(_parse(r["propeptide_coordinates"]))
        if n:
            tcnt[r["genus"]] = tcnt.get(r["genus"], 0) + n

    d["train_count"] = d.genus.map(lambda g: tcnt.get(g, 0))
    d["ab"] = pd.cut(d.train_count, bins=ABINS, labels=ALBL, right=True)
    d["ib"] = pd.cut(d.max_identity_to_train, bins=IBINS, labels=ILBL, right=False)
    return d


def boot(x, B=2000):
    """Mean with a 95 % percentile bootstrap CI (same seed/draw order as the report)."""
    x = np.asarray(x, float)
    n = len(x)
    pt = x.mean()
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return pt, *np.percentile(bs, [2.5, 97.5]), n


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return np.corrcoef(rx, ry)[0, 1]


def spread(values, y0, y1, height_in, pad_pt=3.0, size_pt=6.5):
    """Vertical label positions at least one line apart, in data units.

    Ported from ``paper/fig_tolerance.py`` (the helper is local to that file).
    The two ESM-C curves in panel (b) end 0.04 apart on a 0--1 axis, so their
    end-of-line delta labels were less than one line of type apart. Greedily
    push labels apart from the top down, keeping each inside the axis, and
    return the nudged y for every input value (order preserved).
    """
    span = y1 - y0
    gap = (size_pt + pad_pt) / (height_in * 72.0) * span   # one line, in data
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = [0.0] * len(values)
    prev = None
    for i in order:
        y = values[i]
        if prev is not None and prev - y < gap:
            y = prev - gap
        out[i] = y
        prev = y
    low = min(out)
    if low < y0:                       # stack ran off the bottom: slide it up
        out = [y + (y0 - low) for y in out]
    return out


def bin_curve(s, labels, col, minimum=20):
    """Point estimate + CI per bin, skipping bins with fewer than `minimum` segments."""
    xs, pts, los, his = [], [], [], []
    for bi, b in enumerate(labels):
        cell = s[s[col] == b].matched
        if len(cell) >= minimum:
            mm = boot(cell)
            xs.append(bi)
            pts.append(mm[0])
            los.append(mm[1])
            his.append(mm[2])
    return xs, pts, los, his


# ------------------------------------------------------------------ plot ---

def main() -> int:
    d = load()

    # Spearman rho over genera (>= 5 held-out segments, >= 1 training segment).
    rhos = {}
    for m in MODELS:
        s = d[d.model == m].copy()
        g = (s.groupby("genus")
              .agg(recall=("matched", "mean"), n=("matched", "size"),
                   tc=("train_count", "first"))
              .reset_index())
        g = g[(g.n >= 5) & (g.tc >= 1)]
        rhos[m] = spearman(g.tc, g.recall)

    apply()
    # Stacked panels: at textwidth(0.95) = 5.2 in a side-by-side pair leaves
    # ~2.2 in per panel, too narrow for panel (b)'s six identity-bin labels.
    # The appendix does not count towards the page limit, so height is spent
    # instead.  A bottom strip is reserved for the small caveat footnote.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(textwidth(0.95), 4.7),
                                   layout="constrained")
    # rect is (left, bottom, width, height): the bottom strip holds the grey
    # caveat footnote, the layout keeps the full height above it.
    fig.get_layout_engine().set(rect=(0, 0.095, 1, 0.905))

    # --- (a) representation -------------------------------------------------
    for m in MODELS:
        colour, marker, dash = STYLE[m]
        xs, pts, los, his = bin_curve(d[d.model == m], ALBL, "ab")
        ax1.plot(xs, pts, color=colour, marker=marker, linestyle=dash,
                 linewidth=1.3, markersize=4.2, markeredgecolor="white",
                 markeredgewidth=0.35, label=LABEL[m])
        ax1.fill_between(xs, los, his, color=colour, alpha=0.15, linewidth=0)
    for bi, b in enumerate(ALBL):
        nn = len(d[(d.model == MODELS[0]) & (d.ab == b)])
        ax1.text(bi, 0.035, f"n={nn}", ha="center", fontsize=6.5, color=MUTE)
    ax1.set_xticks(range(len(ALBL)))
    ax1.set_xticklabels(ALBL)
    ax1.set_xlim(-0.32, len(ALBL) - 1 + 0.32)
    ax1.set_xlabel("number of training segments in the genus")
    ax1.set_ylabel("recall (share of segments found)")
    # Literal \u03c1 and \u2212, not mathtext: mathtext would render sans-serif at
    # normal weight inside a serif title.
    rho_line = "Spearman \u03c1: " + ", ".join(
        f"{rhos[m]:+.2f}".replace("-", "\u2212") for m in MODELS)
    panel(ax1, "a", "Recall vs. genus representation\n" + rho_line, dx=DX)

    # --- (b) similarity, with the per-model rise across identity -------------
    ends = {}                       # model -> (x of last bin, end value, delta)
    for m in MODELS:
        colour, marker, dash = STYLE[m]
        xs, pts, los, his = bin_curve(d[d.model == m], ILBL, "ib")
        ax2.plot(xs, pts, color=colour, marker=marker, linestyle=dash,
                 linewidth=1.3, markersize=4.2, markeredgecolor="white",
                 markeredgewidth=0.35, label=LABEL[m])
        ax2.fill_between(xs, los, his, color=colour, alpha=0.15, linewidth=0)
        if len(pts) >= 2:
            ends[m] = (xs[-1], pts[-1], pts[-1] - pts[0])
    for bi, b in enumerate(ILBL):
        nn = len(d[(d.model == MODELS[0]) & (d.ib == b)])
        ax2.text(bi, 0.035, f"n={nn}", ha="center", fontsize=6.5, color=MUTE)
    ax2.set_xticks(range(len(ILBL)))
    ax2.set_xticklabels(ILBL)
    ax2.set_xlim(-0.35, len(ILBL) - 1 + 1.05)   # room for the delta labels
    ax2.set_xlabel("max. segment identity to training data")
    ax2.set_ylabel("recall (share of segments found)")
    panel(ax2, "b", "Recall vs. segment similarity to training data\n"
                    "(\u0394 = gain from least to most similar)", dx=DX)

    for ax in (ax1, ax2):
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", alpha=0)      # categorical axis: no vertical rules
        ax.tick_params(length=0)

    # The three labels are long, so the legend goes inside panel (a), in the
    # empty band below every curve and above the grey n= annotations.
    ax1.legend(loc="lower left", bbox_to_anchor=(0.005, 0.08), fontsize=7,
               handlelength=2.6, labelspacing=0.4, borderaxespad=0.0)
    # Caveat, not a title. First line carries what the deleted suptitle used
    # to say about the intervals; the caption in main.tex does not define them.
    fig.text(0.5, 0.012,
             "shaded bands: 95 % bootstrap CIs; all segments, "
             "pooled held-out folds 2 and 5\n"
             "identity is measured between individual segments, "
             "not whole proteins\n"
             "(the 30 % homology split operates at the protein level)",
             ha="center", va="bottom", fontsize=6.5, color=MUTE,
             linespacing=1.45)
    fig.align_ylabels((ax1, ax2))

    # End-of-line delta labels for panel (b), de-overlapped. Drawn last: the
    # vertical gap one line of type needs, in data units, depends on the final
    # axes height, which constrained layout only fixes once everything else is
    # in place. The white bbox keeps the horizontal grid out of the glyphs.
    fig.canvas.draw()
    h_in = ax2.get_window_extent().height / fig.dpi
    lo, hi = ax2.get_ylim()
    keys = list(ends)
    ys = spread([ends[m][1] for m in keys], lo + 0.02, hi, h_in)
    for m, y_lab in zip(keys, ys):
        x_end, _, delta = ends[m]
        ax2.annotate(f"\u0394 = {delta:+.02f}".replace("-", "\u2212"),
                     (x_end, y_lab), textcoords="offset points", xytext=(5, 0),
                     fontsize=6.5, color=STYLE[m][0], fontweight="bold",
                     ha="left", va="center",
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    finish(fig, "similarity")

    print("rho:", {m: round(rhos[m], 2) for m in MODELS})
    print("abundance-bin n:",
          {b: int(len(d[(d.model == MODELS[0]) & (d.ab == b)])) for b in ALBL})
    print("identity-bin n:",
          {b: int(len(d[(d.model == MODELS[0]) & (d.ib == b)])) for b in ILBL})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
