"""Shared figure style for the paper.

Every figure in the paper should go through `apply()` so that the main-text
and appendix figures are visually one family: AIRI palette, Times to match the
NeurIPS body font, no top/right spines, horizontal grid only.

Usage
-----
    from paperstyle import apply, SERIES_STYLE, textwidth
    apply()
    fig, ax = plt.subplots(figsize=(textwidth(0.95), 2.2), layout="constrained")

Colour is never the only channel: `SERIES_STYLE` pairs each colour with a
distinct marker and dash pattern so the figures survive greyscale printing and
common colour-vision deficiencies.
"""
import sys, pathlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from airi_palette import AIRI, GRID, INK, LINE_CYCLE  # noqa: E402

# NeurIPS 2026 single-column text width, in inches.
TEXTWIDTH_IN = 5.5


def textwidth(frac=1.0):
    """Figure width in inches for a given fraction of \\linewidth."""
    return TEXTWIDTH_IN * frac


# Canonical (colour, marker, linestyle) per role, reused across figures so a
# given model looks the same wherever it appears.
SERIES_STYLE = {
    "base_esm2":       (AIRI["E22"], "o", "-"),
    "base_esmc":       (AIRI["A"],   "s", "--"),
    "boundary":        (AIRI["E21"], "^", "-"),
    "adapter":         (AIRI["F2"],  "D", "-."),
    "full":            (AIRI["C"],   "v", (0, (3, 1, 1, 1))),
    # ESM-C 6B variants (branch B: appear only once runs/5cv_esmc6b_* exist).
    # Same marker as the ESM-2 counterpart, dashed like the ESM-C base.
    "boundary_esmc":   (AIRI["E21"], "^", "--"),
    "adapter_esmc":    (AIRI["F2"],  "D", "--"),
    "full_esmc":       (AIRI["C"],   "v", "--"),
}


def apply(base_font_size=8):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": base_font_size,
        "axes.labelsize": base_font_size,
        "axes.titlesize": base_font_size + 0.5,
        "legend.fontsize": base_font_size - 1,
        "xtick.labelsize": base_font_size - 0.5,
        "ytick.labelsize": base_font_size - 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=LINE_CYCLE),
    })
