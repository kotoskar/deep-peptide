#!/usr/bin/env python3
"""Figure 1: what a boundary error looks like, and how the models separate as the
match tolerance tightens.

Panel (a) draws ONE real precursor from the nested-CV test predictions: the
annotation, the baseline's segmentation and the segmentation of the model with
both additions, with the +-3 acceptance window around each annotated cleavage
site shaded. It is the concrete version of the claim panels (b)/(c) make in
aggregate: the baseline's segments sit inside the +-3 window but not on the
residue, so they are scored as found at the headline tolerance and as misses at
an exact match.

Nothing here is hand-typed: the example's coordinates are read back out of
runs/<model>/<cell>/segments.json.gz, the same artifacts the metrics come from.

Input : runs/5cv_*/nested_cv_tolerance.json  (tolerance_sweep_cv.py)
        runs/5cv_*/<cell>/segments.json.gz   (tolerance_sweep_cv.py)
        data/uniprot_2026/labeled_sequences.csv (for the precursor length)
Output: <outdir>/fig_tolerance.png  (+ .pdf)

Usage: env/bin/python analysis/metrics/src/generators/fig_hook_cv.py
"""
from __future__ import annotations
import argparse, gzip, json, pathlib, sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paperstyle import apply, SERIES_STYLE, textwidth  # noqa: E402
from airi_palette import AIRI, INK  # noqa: E402

MODELS = [
    ("5cv_baseline_esm2_fp32",     "ESM-2, base",                     "base_esm2"),
    ("5cv_esm2_boundary",     "ESM-2 + boundary head",           "boundary"),
    ("5cv_esm2_adapter_only", "ESM-2 + adapter",                 "adapter"),
    ("5cv_esm2_full",         "ESM-2 + boundary head + adapter", "full"),
    ("5cv_esmc6b_plain",      "ESM-C 6B, base",                  "base_esmc"),
    # Picked up automatically once the ESM-C 6B variants land.
    ("5cv_esmc6b_boundary",     "ESM-C 6B + boundary head",           "boundary_esmc"),
    ("5cv_esmc6b_adapter_only", "ESM-C 6B + adapter",                 "adapter_esmc"),
    ("5cv_esmc6b_full",         "ESM-C 6B + boundary head + adapter", "full_esmc"),
]

# The worked example. A frog antimicrobial-peptide precursor whose propeptide and
# mature peptide are adjacent, so both of the baseline's errors fall on cleavage
# sites rather than on the ends of the chain.
EX_CELL = "outer1_inner0"
EX_PROT = "G3ETQ3"
EX_BASE = "5cv_baseline_esm2_fp32"
EX_FULL = "5cv_esm2_full"

TYPE_COLOR = {"peptides": AIRI["E1"], "propeptides": AIRI["B"]}
TYPE_LABEL = {"peptides": "peptide", "propeptides": "propeptide"}
TOL = 3


# ------------------------------------------------------------------ loading ---

def load_summary(name):
    path = pathlib.Path("runs") / name / "nested_cv_tolerance.json"
    return json.load(open(path)) if path.exists() else None


def series(summary, tolerances, task="all"):
    return ([summary[f"cv_tol{t}_{task}_f1_mean"] for t in tolerances],
            [summary[f"cv_tol{t}_{task}_f1_std"] for t in tolerances])


def load_example():
    """Annotation and both segmentations for the worked example, from disk."""
    out = {}
    for key, model in (("base", EX_BASE), ("full", EX_FULL)):
        path = pathlib.Path("runs") / model / EX_CELL / "segments.json.gz"
        with gzip.open(path, "rt") as fh:
            rec = next(r for r in json.load(fh) if r["name"] == EX_PROT)
        out[key] = {t: [tuple(x) for x in rec[t]["pred"]] for t in TYPE_COLOR}
        out["true"] = {t: [tuple(x) for x in rec[t]["true"]] for t in TYPE_COLOR}
    seqs = pd.read_csv("data/uniprot_2026/labeled_sequences.csv", index_col="protein_id")
    out["length"] = len(seqs.loc[EX_PROT, "sequence"])
    return out


# ------------------------------------------------------------------ drawing ---

def draw_example(ax, ex):
    """Three stacked lanes, cropped to the informative part of the chain.

    The panel starts a few residues before the first annotated segment rather
    than at residue 1: the N-terminal stretch carries no segments in any of the
    three lanes and would otherwise spend a third of the width. The leftmost
    tick states where the drawing begins, so the axis is not broken, only
    shortened.
    """
    L = ex["length"]
    lanes = [("annotation", ex["true"], True),
             ("baseline", ex["base"], False),
             ("ours", ex["full"], False)]
    h, gap = 0.9, 0.12                      # lanes essentially touching
    cuts = sorted({c for segs in ex["true"].values() for s in segs for c in s})
    x0 = max(1, min(cuts) - 9)              # keep a little N-terminal context
    top = len(lanes) * (h + gap) - gap

    for c in cuts:                          # +-3 acceptance window
        ax.add_patch(Rectangle((c - TOL - 0.5, 0), 2 * TOL + 1, top,
                               facecolor="#9AA5A6", alpha=0.22, edgecolor="none",
                               zorder=0))
        ax.plot([c, c], [0, top], color="#7F8B8C", lw=0.6, ls=(0, (2, 2)), zorder=4)

    for i, (label, segs, is_true) in enumerate(lanes):
        y = (len(lanes) - 1 - i) * (h + gap)
        ax.add_patch(Rectangle((x0, y), L - x0 + 0.5, h, facecolor="#EFEDE9",
                               edgecolor="#D4D0CA", lw=0.5, zorder=2))
        for task, spans in segs.items():
            for s, e in spans:
                ax.add_patch(Rectangle((s - 0.5, y), e - s + 1, h,
                                       facecolor=TYPE_COLOR[task],
                                       edgecolor="white", lw=0.6, zorder=3))
                if is_true:
                    ax.text((s + e) / 2, y + h / 2, TYPE_LABEL[task], ha="center",
                            va="center", fontsize=6, color="white", zorder=5)
        ax.text(x0 + 0.6, y + h / 2, label, ha="left", va="center", fontsize=6,
                color="#4A4A4A", zorder=6)

    # Displacement of each baseline cut, drawn inside its own lane.
    y_base = (len(lanes) - 2) * (h + gap)
    for task, spans in ex["base"].items():
        for (ps, pe) in spans:
            for (ts, te) in ex["true"][task]:
                if abs(ps - ts) > TOL or abs(pe - te) > TOL:
                    continue
                for pred_c, true_c in ((ps, ts), (pe, te)):
                    d = pred_c - true_c
                    if d == 0:
                        continue
                    ax.annotate("", xy=(pred_c, y_base + h / 2),
                                xytext=(true_c, y_base + h / 2),
                                arrowprops=dict(arrowstyle="-|>", color=AIRI["E21"],
                                                lw=0.9, shrinkA=0, shrinkB=0),
                                zorder=8)
                    ax.text(true_c + (3.4 if d > 0 else -3.4), top + 0.05, f"{d:+d}",
                            ha="center", va="bottom", fontsize=5.8, color=AIRI["E21"])

    ax.set_xlim(x0 - 0.5, L + 1.5)
    ax.set_ylim(-0.35, top + 0.75)
    ax.set_yticks([])
    ax.set_xticks([x0] + [t for t in (20, 30, 40, 50, 60, L) if t > x0 + 3])
    ax.tick_params(axis="x", pad=1.5, length=2)
    ax.grid(False)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(x0, L + 0.5)
    ax.spines["bottom"].set_position(("outward", 2))
    ax.set_title("(a) baseline cuts fall inside the $\\pm3$ window, not on the residue",
                 loc="left", pad=7)


def _trim_white(img, thresh=0.985):
    """Drop the scheme's white margin, which otherwise eats half of panel (b)."""
    a = img[..., :3] / (255.0 if img.dtype != float else 1.0)
    ink = (a.mean(axis=2) < thresh)
    rows, cols = ink.any(axis=1).nonzero()[0], ink.any(axis=0).nonzero()[0]
    if not len(rows) or not len(cols):
        return img
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return img[r0:r1, c0:c1]


def draw_curves(ax_abs, ax_gap, loaded, tol, task):
    x = list(range(len(tol)))
    for _, label, key, s in loaded:
        colour, marker, dash = SERIES_STYLE[key]
        mean, std = series(s, tol, task)
        ax_abs.errorbar(x, mean, yerr=std, label=label, color=colour, marker=marker,
                        linestyle=dash, markersize=3.2, linewidth=1.3,
                        capsize=2, elinewidth=0.7)
    ref_entry = next(((n, l, k, sm) for n, l, k, sm in loaded
                      if n == "5cv_baseline_esm2_fp32"), None)
    if ref_entry is not None:
        _, ref_label, ref_key, ref_summary = ref_entry
        rmean, _ = series(ref_summary, tol, task)
        # The reference is drawn as a series of its own rather than as a bare
        # rule: it is the model every other line is measured against, and a
        # legend entry for a line that is not on the plot is worse than useless.
        colour, marker, dash = SERIES_STYLE[ref_key]
        ax_gap.plot(x, [0] * len(x), color=colour, marker=marker, linestyle=dash,
                    markersize=3.2, linewidth=1.3, label=f"{ref_label} (reference)",
                    zorder=2)
        for name, label, key, sm in loaded:
            if name == "5cv_baseline_esm2_fp32":
                continue
            colour, marker, dash = SERIES_STYLE[key]
            mean, _ = series(sm, tol, task)
            ax_gap.plot(x, [m - r for m, r in zip(mean, rmean)], color=colour,
                        marker=marker, linestyle=dash, markersize=3.2, linewidth=1.3,
                        label=label)
    xlabels = [f"$\\pm{t}$" if t else "exact" for t in tol]
    for ax in (ax_abs, ax_gap):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("boundary-match tolerance")
    ax_abs.set_ylabel("segment F1")
    ax_gap.set_ylabel("F1 gap to the base")
    ax_abs.set_title("(b) absolute F1", loc="left")
    ax_gap.set_title("(c) gap to the base", loc="left")


# --------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="texs/ai4dd/figures")
    ap.add_argument("--tolerances", nargs="*", type=int, default=[3, 2, 1, 0])
    ap.add_argument("--task", default="all", choices=["all", "peptides", "propeptides"])
    ap.add_argument("--no-example", action="store_true",
                    help="draw only the two curve panels (the previous figure)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="also plot runs whose nested CV is not finished (off by default: a "
                         "partial run has no protocol-valid mean)")
    ap.add_argument("--abc", action="store_true",
                    help="problem / architecture / result layout: the precursor example on top, "
                         "the architecture scheme and the gap curve below. The absolute-F1 panel "
                         "drops out, since Table 1 carries the same numbers.")
    ap.add_argument("--arch", default="texs/ai4dd/figures/architecture_scheme.jpg")
    ap.add_argument("--esm2-only", action="store_true",
                    help="drop the ESM-C series, to match a main text that parks them")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loaded = []
    models = [m for m in MODELS if not (args.esm2_only and "esmc" in m[0])]
    for name, label, key in models:
        s = load_summary(name)
        if s is None:
            print(f"[skip] {name}: no nested_cv_tolerance.json yet")
            continue
        if not s.get("complete", False):
            # A partial run has no protocol-valid mean -- its outer folds carry
            # different numbers of inner cells -- so it stays off the figure
            # unless it is asked for explicitly.
            if not args.allow_incomplete:
                print(f"[skip] {name}: INCOMPLETE ({s['n_cells']}/20 cells)")
                continue
            print(f"[warn] {name}: INCOMPLETE ({s['n_cells']}/20 cells), plotted on request")
        loaded.append((name, label, key, s))
    if not loaded:
        print("nothing to plot")
        return 1

    apply()
    if args.no_example:
        fig, (ax_abs, ax_gap) = plt.subplots(1, 2, figsize=(textwidth(1.0), 2.35),
                                             layout="constrained")
    elif args.abc:
        # The scheme is 2.24:1, so it needs the wider of the two lower cells or it
        # sits in a letterbox with the curve panel towering over it.
        fig = plt.figure(figsize=(textwidth(1.0), 3.02), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[0.52, 1.5], width_ratios=[1.12, 1.0],
                              hspace=0.02, wspace=0.05)
        ax_ex = fig.add_subplot(gs[0, :])
        ax_arch = fig.add_subplot(gs[1, 0])
        ax_gap = fig.add_subplot(gs[1, 1])
        ex = load_example()
        draw_example(ax_ex, ex)
        ax_arch.imshow(_trim_white(mpimg.imread(args.arch)))
        ax_arch.set_anchor("W")
        ax_arch.set_axis_off()
        ax_arch.set_title("(b) where they attach", loc="left")
    else:
        fig = plt.figure(figsize=(textwidth(1.0), 3.02), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[0.58, 1.5], hspace=0.02)
        ax_ex = fig.add_subplot(gs[0, :])
        ax_abs = fig.add_subplot(gs[1, 0])
        ax_gap = fig.add_subplot(gs[1, 1])
        ex = load_example()
        draw_example(ax_ex, ex)
        print(f"[example] {EX_PROT} ({EX_CELL}) length {ex['length']}: "
              f"true {ex['true']} base {ex['base']} full {ex['full']}")

    if args.abc:
        # draw_curves wants two axes; give it a throwaway for the absolute panel
        # and keep only the gap, which is the panel that carries the argument.
        scratch = fig.add_subplot(gs[1, 0]); scratch.set_visible(False)
        draw_curves(scratch, ax_gap, loaded, args.tolerances, args.task)
        ax_gap.set_ylabel("")
        ax_gap.set_title("(c) F1 gap to the ESM-2 base", loc="left")
        legend_src = ax_gap
    else:
        draw_curves(ax_abs, ax_gap, loaded, args.tolerances, args.task)
        legend_src = ax_abs
    handles, labels = legend_src.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               handlelength=2.4, columnspacing=1.6)

    png = outdir / "fig_tolerance.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
