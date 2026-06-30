"""Canonical model labels + colors + presentation helpers, shared by all figure
generators so legends/colours are unified (full modification chain, as on the
F1-by-length figure). Import: from _pres import MODEL_LABELS, COLORS, ...

Presentation mode: set env PRES=1 -> figures go to presentation/figures/ only,
titles drop the parenthetical/(а)(б) chrome, gray helper annotations are suppressed.
When PRES is unset, generators keep their original report behaviour unchanged.
"""
import os, re
PRES = os.environ.get("PRES") == "1"

# Full modification chain spelled out (the "полное описание модификаций" the user wants).
MODEL_LABELS = {
    "baseline_esm2":               "ESM-2 (бейзлайн)",
    "esmc_600m":                   "ESM-C 600M",
    "esmc_6b":                     "ESM-C 6B",
    "esmc6b_boundary":             "ESM-C 6B + boundary",
    "adapter256":                  "ESM-C 6B + boundary + gated(256)",
    "esmc6b_3di_gated_boundary":   "ESM-C 6B + boundary + gated(256) + 3Di",
    "esmc6b_3di_nocompress":       "ESM-C 6B + boundary + gated(2560) + 3Di",
    "esmc6b_3di_zeroctrl":         "ESM-C 6B + boundary + gated(256), 3Di занулён",
    "esmc6b_boundary_bond":        "ESM-C 6B + boundary + bond-лосс",
}
COLORS = {
    "baseline_esm2":               "#9aa3ad",
    "esmc_600m":                   "#8fb0d6",
    "esmc_6b":                     "#5b8bc0",
    "esmc6b_boundary":             "#e0913f",
    "adapter256":                  "#9a78c2",
    "esmc6b_3di_gated_boundary":   "#4ca37a",
    "esmc6b_3di_nocompress":       "#5fae93",
    "esmc6b_3di_zeroctrl":         "#9a78c2",
    "esmc6b_boundary_bond":        "#c25a5a",
}
# segment-type colours (consistent with data_distributions)
PEP_COLOR, PROPEP_COLOR = "#4c9be8", "#e0913f"

RC = {"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":13,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":10,
 "axes.labelsize":11,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":10}
INK = "#1a1a1a"

MUTE = "#8a9099"

def clean_title(s):
    """Strip presentation title chrome: the leading '(а)/(б)' panel tag, ANY
    parenthetical caption, and the trailing 'объединение … фолдов {…}' clause."""
    s = re.sub(r"^\([абвг]\)\s*", "", s)                          # leading panel tag
    s = re.sub(r"\([^)]*\)", "", s)                               # any parenthetical
    s = re.sub(r"[,\s]*(?:на|в|по)?\s*объединени\w*[^,\n]*", "", s)  # fold-union clause
    s = re.sub(r"\s*\n\s*", " ", s)                               # join wrapped lines
    s = re.sub(r"\s+([,:])", r"\1", s)                            # tidy ' ,' / ' :'
    s = re.sub(r"\s{2,}", " ", s).strip(" ,\n")
    return s

def title(s):
    """Presentation strips the title chrome; report keeps it unchanged."""
    return clean_title(s) if PRES else s

def outdirs():
    """Where to write a figure: presentation-only when PRES, else the report dirs."""
    if PRES:
        return ["presentation/figures/"]
    return ["analysis/metrics/figures/", "texs/Overleaf/figures/"]
