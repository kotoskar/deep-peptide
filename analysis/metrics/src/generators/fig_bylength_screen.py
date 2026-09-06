#!/usr/bin/env python3
"""Segment F1 against segment length under the earlier seven-fold screening split.

This is the screening-protocol ancestor of the nested-CV figure: the same
question, asked on the split that first raised it. Quality depends strongly on
how long the true segment is, the two held-out folds carry different length
profiles, and that combination is what made the single-split effect sizes flip
sign between folds.

    env/bin/python analysis/metrics/src/generators/fig_bylength_screen.py
"""
from __future__ import annotations
import argparse, pathlib, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paperstyle import apply, SERIES_STYLE, textwidth  # noqa: E402

FOLDS = [2, 5]
EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]
BINS = [f"{EDGES[i]}-{EDGES[i + 1] - 1}" for i in range(len(EDGES) - 1)]
MIN_TRUE = 25  # bins thinner than this are not scored, as in the screening figures

SHOW = [
    ("baseline_esm2",   "ESM-2, base",               "base_esm2"),
    ("esmc_6b",         "ESM-C 6B, base",            "base_esmc"),
    ("esmc6b_boundary", "ESM-C 6B + boundary head",  "boundary_esmc"),
]


def lbin(length):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= length < EDGES[i + 1]:
            return BINS[i]
    return None


def f1(tp, fn, fp):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/metrics")
    ap.add_argument("--outdir", default="texs/ai4dd/figures")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    t = pd.read_csv(root / "clean_tol_true.csv")
    p = pd.read_csv(root / "clean_tol_pred.csv")
    t = t[t.fold.isin(FOLDS)]
    p = p[p.fold.isin(FOLDS)]

    # Score every model on the proteins all of them cover, so the lines differ
    # by model and not by which proteins each run happened to see.
    def cover(m):
        s = t[t.model == m]
        return set(zip(s.fold, s.protein))
    common = set.intersection(*[cover(m) for m, _, _ in SHOW])

    def keep(df):
        return df[[(f, pr) in common for f, pr in zip(df.fold, df.protein)]]

    t, p = keep(t), keep(p)
    t = t.assign(lb=t.length.map(lbin))
    p = p.assign(lb=p.plen.map(lbin))

    curves = {}
    for m, _, _ in SHOW:
        tm, pm = t[t.model == m], p[p.model == m]
        out = {}
        for b in BINS:
            tb, pb = tm[tm.lb == b], pm[pm.lb == b]
            if len(tb) < MIN_TRUE:
                continue
            out[b] = f1(tb.m3.sum(), (tb.m3 == 0).sum(), (pb.m3 == 0).sum())
        curves[m] = out

    kept = [b for b in BINS if all(b in curves[m] for m, _, _ in SHOW)]
    x = list(range(len(kept)))

    apply()
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(textwidth(1.0), 2.9), height_ratios=[2.6, 1.0],
        sharex=True, layout="constrained")

    for m, label, key in SHOW:
        colour, marker, dash = SERIES_STYLE[key]
        ax.plot(x, [curves[m][b] for b in kept], color=colour, marker=marker,
                linestyle=dash, markersize=3.4, linewidth=1.3, label=label)

    # What each held-out fold is made of, on that same axis.
    base = t[t.model == SHOW[0][0]]
    width = 0.38
    for i, (f, shade) in enumerate(zip(FOLDS, ["#9AA5A6", "#C9D1D2"])):
        counts = [int(((base.fold == f) & (base.lb == b)).sum()) for b in kept]
        axn.bar([xi + (i - 0.5) * width for xi in x], counts, width=width,
                color=shade, linewidth=0, label=f"fold {f}")

    ax.set_ylabel("segment F1 at $\\pm3$")
    axn.set_ylabel("true segments\nin the fold")
    axn.set_xticks(x)
    axn.set_xticklabels([b.replace("-", "--") for b in kept])
    axn.set_xlabel("segment length (residues)")
    for a in (ax, axn):
        a.margins(x=0.04)
    fig.legend(loc="outside lower center", ncols=3, handlelength=2.4, columnspacing=1.6)

    out = pathlib.Path(args.outdir) / "fig_bylength_screen.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (common proteins: {len(common)})")

    for m, label, _ in SHOW:
        ys = [curves[m][b] for b in kept]
        lo, hi = min(ys), max(ys)
        print(f"{label:26s} F1 {lo:.3f} ({kept[ys.index(lo)]}) .. {hi:.3f} "
              f"({kept[ys.index(hi)]}), spread {hi - lo:.3f}")
    for f in FOLDS:
        sub = base[base.fold == f]
        share = [round(float(((sub.lb == b).sum()) / len(sub)), 3) for b in kept]
        print(f"fold {f} share by bin: {dict(zip(kept, share))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
