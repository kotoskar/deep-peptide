#!/usr/bin/env python3
"""Appendix figure: F1 / precision / recall by segment length, under nested CV.

The nested-CV counterpart of the single-split by-length figure
(`fig_bylength.py`). Same three panels and the same length bins, but the five
configurations that actually have a 5x4 nested-CV grid, and error bars that mean
what they mean everywhere else in this paper: the standard deviation across the
5 outer folds.

Colours, markers, dashes and legend labels come from `paperstyle.SERIES_STYLE`,
the Figure-1 family, so a configuration looks the same here as it does in
Figure 1 -- unlike the single-split appendix figures, which draw a different set
of models.

Reads  analysis/metrics/bylength_cv.json   (analysis/metrics/src/bylength_cv.py)
Writes texs/ai4dd/figures/bylength_cv.png  at textwidth(0.95)

Usage: env/bin/python analysis/metrics/src/generators/paper/fig_bylength_cv.py
"""
from __future__ import annotations

import json
import pathlib

from _common import apply, finish, panel, plt, textwidth
from paperstyle import SERIES_STYLE

SRC = pathlib.Path("analysis/metrics/bylength_cv.json")

# run name -> (Figure-1 legend label, SERIES_STYLE key)
MODELS = [
    ("5cv_baseline_esm2_fp32",     "ESM-2, base",                     "base_esm2"),
    ("5cv_esm2_boundary",     "ESM-2 + boundary head",           "boundary"),
    ("5cv_esm2_adapter_only", "ESM-2 + adapter",                 "adapter"),
    ("5cv_esm2_full",         "ESM-2 + boundary head + adapter", "full"),
    ("5cv_esmc6b_plain",      "ESM-C 6B, base",                  "base_esmc"),
    # Branch B: drawn automatically once the ESM-C 6B variants land.
    ("5cv_esmc6b_boundary",     "ESM-C 6B + boundary head",           "boundary_esmc"),
    ("5cv_esmc6b_adapter_only", "ESM-C 6B + adapter",                 "adapter_esmc"),
    ("5cv_esmc6b_full",         "ESM-C 6B + boundary head + adapter", "full_esmc"),
]

PANELS = [("a", "F1", "f1"), ("b", "Precision", "precision"), ("c", "Recall", "recall")]


def main() -> int:
    data = json.loads(SRC.read_text())
    bins = data.get("kept_bins") or data["bins"]
    present = [(n, lab, key) for n, lab, key in MODELS if n in data["models"]]
    if not present:
        print("nothing to plot")
        return 1
    for name, _, _ in MODELS:
        if name not in data["models"]:
            print(f"[skip] {name}: no nested-CV run yet")

    apply()
    fig, axes = plt.subplots(1, 3, figsize=(textwidth(0.95), 2.8),
                             sharey=True, layout="constrained")
    x = list(range(len(bins)))
    # Five overlapping error bars on one x position are unreadable; nudging each
    # series sideways by a fraction of a bin separates them without moving any
    # value. The nudge is symmetric about the bin, so no series is favoured.
    dodge = 0.13
    offsets = [(i - (len(present) - 1) / 2) * dodge for i in range(len(present))]

    for ax, (letter, title, metric) in zip(axes, PANELS):
        for (name, label, key), off in zip(present, offsets):
            m = data["models"][name]["metrics"]
            colour, marker, dash = SERIES_STYLE[key]
            mean = [m[f"{b}_{metric}"]["mean"] for b in bins]
            std = [m[f"{b}_{metric}"]["std"] for b in bins]
            ax.errorbar([xi + off for xi in x], mean, yerr=std, label=label,
                        color=colour, marker=marker, linestyle=dash,
                        markersize=2.8, linewidth=1.1, capsize=1.2,
                        elinewidth=0.55, alpha=0.95)
        panel(ax, letter, title, dx=-0.16)
        ax.set_xticks(x)
        # The bin's lower edge is enough of a tick label to stay horizontal at
        # this width; the shared axis label says the bins are five residues wide.
        ax.set_xticklabels([b.split("-")[0] for b in bins])
        # Trimmed from the full [0, 1]: every curve and every whisker lives
        # inside this band, and the model differences the figure exists to show
        # are 0.03-0.09 wide, which a full-range axis flattens.
        ax.set_ylim(0.05, 0.90)
        ax.grid(axis="x", alpha=0)
    axes[0].set_ylabel("metric value")
    # One axis label for the row, on the middle panel: three copies collide at
    # this width, and fig.supxlabel would land under the outside legend.
    axes[1].set_xlabel("segment length (5-residue bins)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncols=2, handlelength=2.4, columnspacing=1.6)

    finish(fig, "bylength_cv")

    for name, label, _ in present:
        m = data["models"][name]["metrics"]
        line = "  ".join(f"{b}:{m[f'{b}_f1']['mean']:.3f}" for b in bins)
        print(f"  {label:34s} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
