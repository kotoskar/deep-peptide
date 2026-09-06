#!/usr/bin/env python3
"""Per-outer-fold scores for every configuration, on one axis.

The paper argues that the folds are not exchangeable and that their spread
exceeds the effects measured on them. That is an argument about a shape: every
configuration rises and falls together across the folds, so the fold explains
more of the variation than the architecture does. A table of means cannot show
it and a table of per-fold numbers makes the reader do the work, so this draws
it: eight lines, five folds, and the vertical spread of one line against the
vertical gap between lines.

    env/bin/python analysis/metrics/src/generators/fig_perfold.py
"""
from __future__ import annotations
import argparse, json, pathlib, re, statistics as st, sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paperstyle import apply, SERIES_STYLE, textwidth  # noqa: E402
from airi_palette import INK  # noqa: E402

RUNS = [
    ("5cv_baseline_esm2_fp32", "ESM-2, base", "base_esm2"),
    ("5cv_esm2_boundary",      "ESM-2 + boundary head", "boundary"),
    ("5cv_esm2_adapter_only",  "ESM-2 + adapter", "adapter"),
    ("5cv_esm2_full",          "ESM-2 + both", "full"),
    ("5cv_esmc6b_plain",       "ESM-C 6B, base", "base_esmc"),
    ("5cv_esmc6b_boundary",    "ESM-C 6B + boundary head", "boundary_esmc"),
    ("5cv_esmc6b_adapter_only", "ESM-C 6B + adapter", "adapter_esmc"),
    ("5cv_esmc6b_full",        "ESM-C 6B + both", "full_esmc"),
]


def fold_median_lengths(labels_csv, split_csv):
    """Median true-segment length in each outer fold.

    The folds are balanced on homology and cleavage motif, not on segment
    length, so this axis was left free, and it is the axis quality tracks.
    """
    lab = pd.read_csv(labels_csv)
    sp = pd.read_csv(split_csv)
    fold = dict(zip(sp["AC"], sp["cluster"]))
    lab = lab[lab["protein_id"].isin(fold)]
    lens = {}
    for _, r in lab.iterrows():
        f = int(fold[r["protein_id"]])
        for col in ("coordinates", "propeptide_coordinates"):
            v = r[col]
            if isinstance(v, str):
                for a, b in re.findall(r"\((\d+)-(\d+)\)", v):
                    lens.setdefault(f, []).append(int(b) - int(a) + 1)
    return {f: st.median(v) for f, v in lens.items()}


def per_outer(run, root, tol=3):
    d = json.load(open(pathlib.Path(root) / run / "nested_cv_tolerance.json"))
    v = d[f"per_outer_tol{tol}_all_f1"]
    return [v[k] for k in sorted(v, key=int)] if isinstance(v, dict) else list(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--outdir", default="texs/ai4dd/figures")
    ap.add_argument("--labels", default="data/uniprot_2026/labeled_sequences.csv")
    ap.add_argument("--split",
                    default="data/uniprot_2026/graphpart_assignments_5motif.esm2covered.csv")
    args = ap.parse_args()

    series = []
    for run, label, key in RUNS:
        try:
            series.append((label, key, per_outer(run, args.runs)))
        except FileNotFoundError:
            print(f"[skip] {run}")

    # What each fold is made of, on the axis quality actually tracks.
    med = fold_median_lengths(args.labels, args.split)

    apply()
    fig, (ax, axl) = plt.subplots(
        2, 1, figsize=(textwidth(1.0), 2.9), height_ratios=[2.4, 1.0],
        sharex=True, layout="constrained")
    x = list(range(5))
    for label, key, ys in series:
        colour, marker, dash = SERIES_STYLE[key]
        ax.plot(x, ys, color=colour, marker=marker, linestyle=dash,
                markersize=3.4, linewidth=1.3, label=label)

    # The band every configuration shares: how much of the picture is the fold.
    lo = [min(ys[i] for _, _, ys in series) for i in x]
    hi = [max(ys[i] for _, _, ys in series) for i in x]
    ax.fill_between(x, lo, hi, color="#9AA5A6", alpha=0.13, zorder=0, linewidth=0)

    axl.bar(x, [med[i] for i in x], color="#9AA5A6", alpha=0.55, width=0.6, linewidth=0)
    axl.set_ylabel("median segment\nlength (residues)")
    axl.set_xticks(x)
    axl.set_xticklabels([f"fold {i}" for i in x])
    axl.set_xlabel("outer fold of the nested cross-validation")
    ax.set_ylabel("segment F1 at $\\pm3$")
    for a in (ax, axl):
        a.margins(x=0.04)
    fig.legend(loc="outside lower center", ncols=2, handlelength=2.4, columnspacing=1.6)

    out = pathlib.Path(args.outdir) / "fig_perfold.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")

    # the same numbers as a table, so the appendix can carry both
    rows = []
    for label, _, ys in series:
        cells = " & ".join(f"${y:.3f}$" for y in ys)
        rows.append(f"{label} & {cells} & ${st.mean(ys):.3f} \\pm {st.stdev(ys):.3f}$ \\\\")
    tex = (out.parent / "perfold_table.tex")
    tex.write_text(
        "\\begin{tabular}{lccccc c}\n\\toprule\n"
        "\\textbf{Configuration} & \\textbf{f0} & \\textbf{f1} & \\textbf{f2} & \\textbf{f3}"
        " & \\textbf{f4} & \\textbf{mean} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {tex}")

    order_f1 = sorted(x, key=lambda i: series[0][2][i])
    order_len = sorted(x, key=lambda i: med[i])
    print(f"folds by base F1     : {order_f1}")
    print(f"folds by median length: {order_len}")
    spread = [max(ys) - min(ys) for _, _, ys in series]
    across = [max(s[2][i] for s in series) - min(s[2][i] for s in series) for i in x]
    print(f"within-configuration range across folds: {min(spread):.3f} to {max(spread):.3f}")
    print(f"across-configuration range within a fold: {min(across):.3f} to {max(across):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
