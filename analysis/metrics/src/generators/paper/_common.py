"""Shared pieces for the English, paper-style appendix figures.

Why this exists
---------------
The report generators one directory up (`report_figs_*.py`, `trades_fig.py`, ...)
render the *Russian* report/presentation figures and must keep doing so. The
AI4DD paper needs the same numbers in English, in the AIRI palette, at true
print size. So the paper figures live here as separate scripts that reuse the
data logic of their Russian counterpart and nothing else.

House style (matches Figure 1, `../fig_hook_cv.py`)
--------------------------------------------------
* `paperstyle.apply()` for fonts/grid/spines -- serif, 8 pt base, no top/right
  spine, horizontal grid only.
* **Draw at true print size**: figure width = `textwidth(frac)` where `frac` is
  the fraction in the figure's `\\MaybeImage{...}{frac}` call in `main.tex`, so
  the PNG lands in the PDF at scale 1 and 8 pt in the figure is 8 pt on paper.
  Height is free: every one of these figures is in the appendix, which does not
  count towards the page limit.
* **No figure-level suptitle.** Each panel carries its own short centred title;
  the panel letter is a separate bold left title, outdented past the y-label
  (`panel()` below). Single-panel figures get one centred title and no letter.
* Colour is never the only channel: `STYLE` pairs each model with a marker and
  a dash pattern.

Run every script from the repo root, e.g.

    env/bin/python analysis/metrics/src/generators/paper/fig_datadist.py
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # ../  -> paperstyle, airi_palette

import matplotlib                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

from airi_palette import AIRI, INK, GRID       # noqa: E402
from paperstyle import apply, textwidth        # noqa: E402

__all__ = [
    "AIRI", "INK", "GRID", "MUTE", "apply", "textwidth",
    "LABEL", "STYLE", "color", "TYPE_COLOR", "TYPE_LABEL",
    "PRECISION_COLOR", "RECALL_COLOR", "F1_COLOR",
    "panel", "finish", "OUTDIR",
]

# Where the paper reads its figures from.
OUTDIR = pathlib.Path("texs/ai4dd/figures")

# Grey for helper annotations (n=..., value labels next to bars).
MUTE = AIRI["E3"]

# ---------------------------------------------------------------- models ---
# Keys are the ones used by the Russian generators (`.._pres.py`), so the data
# logic ports across unchanged; only the label and the style change.
LABEL = {
    "baseline_esm2":             "ESM-2 (baseline)",
    "esmc_600m":                 "ESM-C 600M",
    "esmc_6b":                   "ESM-C 6B",
    "esmc6b_boundary":           "ESM-C 6B + boundary",
    "adapter256":                "ESM-C 6B + boundary + gated(256)",
    "esmc6b_3di_gated_boundary": "ESM-C 6B + boundary + gated(256) + 3Di",
    "esmc6b_3di_nocompress":     "ESM-C 6B + boundary + gated(2560) + 3Di",
    "esmc6b_3di_zeroctrl":       "ESM-C 6B + boundary + gated(256), 3Di zeroed",
    "esmc6b_boundary_bond":      "ESM-C 6B + boundary + bond loss",
}

# (colour, marker, linestyle). The family logic mirrors Figure 1: the ESM-2
# baseline is neutral grey, the ESM-C bases are teal, the boundary head is red,
# the gated adapter violet, and the full 3Di model brown; controls reuse their
# parent's hue one step lighter.
STYLE = {
    "baseline_esm2":             (AIRI["E22"], "o", "-"),
    "esmc_600m":                 (AIRI["B"],   "s", ":"),
    "esmc_6b":                   (AIRI["A"],   "s", "--"),
    "esmc6b_boundary":           (AIRI["E21"], "^", "-"),
    "adapter256":                (AIRI["F2"],  "D", "-."),
    "esmc6b_3di_gated_boundary": (AIRI["C"],   "v", (0, (3, 1, 1, 1))),
    "esmc6b_3di_nocompress":     (AIRI["D1"],  "P", (0, (4, 1.5))),
    "esmc6b_3di_zeroctrl":       (AIRI["F3"],  "d", (0, (1.5, 1.5))),
    "esmc6b_boundary_bond":      (AIRI["E1"],  "X", (0, (5, 2))),
}


def color(key):
    return STYLE[key][0]


# Segment types, identical to Figure 1's `TYPE_COLOR` in ../fig_hook_cv.py.
TYPE_COLOR = {"peptides": AIRI["E1"], "propeptides": AIRI["B"]}
TYPE_LABEL = {"peptides": "peptides", "propeptides": "propeptides"}

# Metric colours for figures whose series are metrics, not models. Deliberately
# not the segment-type hues, so a precision/recall legend cannot be misread as
# peptide/propeptide.
PRECISION_COLOR = AIRI["A"]
RECALL_COLOR = AIRI["F1"]
F1_COLOR = AIRI["E22"]


# ----------------------------------------------------------------- layout ---

def panel(ax, letter=None, title=None, dx=-0.085):
    """Fig.-3 style panel head: bold `(a)` on the left, short title centred.

    `letter` is drawn as the axes' *left* title and pushed out by `dx` (axes
    fractions) so it clears the y tick labels, exactly as in the dataset
    composition figure. Pass `letter=None` on a single-panel figure.
    """
    if letter:
        t = ax.set_title(f"({letter})", loc="left", fontweight="bold")
        t.set_x(dx)
    if title:
        ax.set_title(title, loc="center")


def finish(fig, name, outdir=OUTDIR, pdf=False):
    """Save at true print size and report the width the PDF will see."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    # No bbox_inches="tight": it silently changes the figure width, which is
    # the one thing that must stay equal to textwidth(frac).
    fig.savefig(png)
    if pdf:
        fig.savefig(png.with_suffix(".pdf"))
    w_in = fig.get_size_inches()[0]
    print(f"[fig] {png}  {w_in:.2f} in wide "
          f"({w_in / 5.5:.3f} of \\linewidth), {fig.get_size_inches()[1]:.2f} in tall")
    return png
