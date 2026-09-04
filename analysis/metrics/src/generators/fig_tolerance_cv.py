#!/usr/bin/env python3
"""Figure 1 + the tolerance table, from the nested-CV tolerance sweep.

Input : runs/5cv_*/nested_cv_tolerance.json (written by
        analysis/metrics/src/tolerance_sweep_cv.py)
Output: <outdir>/fig_tolerance.png   -- (a) F1 vs tolerance, (b) absolute gap to the base
        <outdir>/tolerance_table.tex -- booktabs table, F1 at tol 3/2/1/0 + retention
        <outdir>/tolerance_table.csv -- same numbers, machine readable

Error bars / +- are the standard deviation across the 5 outer folds, matching
how the main results table reports uncertainty (the 4 inner models sharing an
outer fold are not independent repeats).

Usage: env/bin/python analysis/metrics/src/generators/fig_tolerance_cv.py \
           [--outdir texs/ai4dd/figures] [--tolerances 3 2 1 0]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import apply, SERIES_STYLE, textwidth

# run name -> (legend label, SERIES_STYLE key)
MODELS = [
    ("5cv_baseline_esm2",     "ESM-2, base",                   "base_esm2"),
    ("5cv_esm2_boundary",     "ESM-2 + boundary head",         "boundary"),
    ("5cv_esm2_adapter_only", "ESM-2 + adapter",               "adapter"),
    ("5cv_esm2_full",         "ESM-2 + boundary head + adapter", "full"),
    ("5cv_esmc6b_plain",      "ESM-C 6B, base",                "base_esmc"),
    # Branch B: picked up automatically once the runs land (missing dirs are skipped).
    ("5cv_esmc6b_boundary",     "ESM-C 6B + boundary head",           "boundary_esmc"),
    ("5cv_esmc6b_adapter_only", "ESM-C 6B + adapter",                 "adapter_esmc"),
    ("5cv_esmc6b_full",         "ESM-C 6B + boundary head + adapter", "full_esmc"),
]


def load(name: str):
    path = pathlib.Path("runs") / name / "nested_cv_tolerance.json"
    if not path.exists():
        return None
    return json.load(open(path))


def series(summary: dict, tolerances, task="all"):
    mean = [summary[f"cv_tol{t}_{task}_f1_mean"] for t in tolerances]
    std = [summary[f"cv_tol{t}_{task}_f1_std"] for t in tolerances]
    return mean, std


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="texs/ai4dd/figures")
    ap.add_argument("--tolerances", nargs="*", type=int, default=[3, 2, 1, 0])
    ap.add_argument("--task", default="all", choices=["all", "peptides", "propeptides"])
    args = ap.parse_args()
    tol = args.tolerances
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for name, label, style_key in MODELS:
        s = load(name)
        if s is None:
            print(f"[skip] {name}: no nested_cv_tolerance.json yet")
            continue
        if not s.get("complete", False):
            print(f"[warn] {name}: INCOMPLETE ({s['n_cells']}/20 cells)")
        loaded.append((name, label, style_key, s))
    if not loaded:
        print("nothing to plot")
        return 1

    apply()
    fig, axes = plt.subplots(1, 2, figsize=(textwidth(1.0), 2.35), layout="constrained")
    x = list(range(len(tol)))
    xlabels = [f"$\\pm{t}$" if t else "exact" for t in tol]

    for _, label, key, s in loaded:
        colour, marker, dash = SERIES_STYLE[key]
        mean, std = series(s, tol, args.task)
        axes[0].errorbar(x, mean, yerr=std, label=label, color=colour, marker=marker,
                         linestyle=dash, markersize=3.2, linewidth=1.3,
                         capsize=2, elinewidth=0.7)
    # Panel (b): absolute gap to the ESM-2 base at each tolerance. A ratio such
    # as "fraction of +-3 F1 retained" moves whenever a model is better overall,
    # so it cannot separate finding more segments from placing them better; the
    # absolute gap can. A flat line is a parallel translation of the baseline,
    # a rising line is a boundary that is genuinely sharper.
    ref = next((s for n, _, _, s in loaded if n == "5cv_baseline_esm2"), None)
    if ref is not None:
        rmean, _ = series(ref, tol, args.task)
        axes[1].axhline(0, color="#9AA5A6", linewidth=0.8, zorder=1)
        for name, label, key, s in loaded:
            if name == "5cv_baseline_esm2":
                continue
            colour, marker, dash = SERIES_STYLE[key]
            mean, _ = series(s, tol, args.task)
            axes[1].plot(x, [m - r for m, r in zip(mean, rmean)], color=colour,
                         marker=marker, linestyle=dash, markersize=3.2, linewidth=1.3)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("boundary-match tolerance")
    axes[0].set_ylabel("segment F1")
    axes[1].set_ylabel("F1 gap to the ESM-2 base")
    axes[0].set_title("(a) absolute F1", loc="left")
    axes[1].set_title("(b) gap to the base", loc="left")
    # One legend under both panels: inside panel (a) it sits on top of the
    # lowest curve at the tight-tolerance end, which is where the plot is busiest.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncols=2, handlelength=2.4, columnspacing=1.6)

    png = outdir / "fig_tolerance.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"[fig] {png}")

    # ------------------------------------------------------------- table ---
    rows = []
    rmean = series(ref, tol, args.task)[0] if ref is not None else None
    for name, label, _, s in loaded:
        mean, std = series(s, tol, args.task)
        gap = (mean[-1] - rmean[-1]) - (mean[0] - rmean[0]) if rmean else float("nan")
        rows.append((label, mean, std, gap))

    tex = [
        r"\begin{tabular}{l" + "c" * len(tol) + "c}",
        r"\toprule",
        "\\textbf{Configuration} & "
        + " & ".join(f"$\\pm{t}$" if t else "exact" for t in tol)
        + r" & \textbf{gap growth} \\",
        r"\midrule",
    ]
    for label, mean, std, gap in rows:
        # Compact form for the main text: the fold-level spread is already in
        # tab:main-results and repeating it here only costs width.
        cells = " & ".join(f"${m:.3f}$" for m in mean)
        tex.append(f"{label} & {cells} & ${gap:+.3f}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_path = outdir / "tolerance_table.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    print(f"[tab] {tex_path}")

    csv = ["config," + ",".join(f"tol{t}_f1,tol{t}_std" for t in tol) + ",gap_growth"]
    for label, mean, std, gap in rows:
        vals = ",".join(f"{m:.4f},{s:.4f}" for m, s in zip(mean, std))
        csv.append(f"\"{label}\",{vals},{gap:+.4f}")
    csv_path = outdir / "tolerance_table.csv"
    csv_path.write_text("\n".join(csv) + "\n")
    print(f"[csv] {csv_path}")

    for label, mean, std, gap in rows:
        span = "  ".join(f"{t}:{m:.3f}" for t, m in zip(tol, mean))
        print(f"  {label:38s} {span}   gap_growth={gap:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
