#!/usr/bin/env python3
"""Per-model recall-by-length figure -> texs/error_analysis/figures/recall_by_length_models.png.

Recreates the generator behind commit f222d53 (which shipped only the PNG). Reads
every per-run segment table analysis/errors/error_stats/<run>__segments.csv
(produced by error_analysis.py) and draws one recall-by-length-bin line per model
for the PEPTIDES task. Any new <run>__segments.csv is picked up automatically, so
adding the ESM-C 6B + boundary winner is just: produce its segments csv, re-run.

Length bins and matching match the pooled figure (recall_by_length.png): ±3
corrected matcher, bins 5 / 6-10 / 11-20 / 21-30 / 31-50.

Usage:
  env/bin/python analysis/errors/src/plot_recall_by_length_models.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
STATS = ROOT / "analysis" / "errors" / "error_stats"
FIG = ROOT / "texs" / "error_analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LEN_LABELS = ["5", "6-10", "11-20", "21-30", "31-50"]
LMIN, LMAX, WIN = 5, 50, 2  # per-integer length range + ±WIN smoothing (as in the stacked plot)

# Short display labels; runs without an entry fall back to the folder name.
LABELS = {
    "train_run_esm2": "ESM2 (baseline)",
    "esm2_telescoping_segmental": "ESM2 telescoping",
    "esm2_aho_mid_fusion_raw_m64": "ESM2 Aho mid-fusion",
    "esm2_aho_emission_fusion": "ESM2 Aho emis-fusion",
    "esm2_aho_emission_fusion_h32": "ESM2 Aho emis-fusion h32",
    "train_run_esm2+3di_proj": "ESM2+3Di proj (raw)",
    "train_run_esm2+3di_proj_gated_conv": "ESM2+3Di gated conv",
    "train_run_esm2_aft_single_gated": "ESM2 AFT single gated",
    "train_run_esmc_600m": "ESM-C 600M",
    "esmc6b_boundary_bond": "ESM-C 6B + boundary (old best)",
    "esmc6b_3di_gated_boundary": "ESM-C 6B ⊕ 3Di gated + boundary ★",
}
WINNER = "esmc6b_3di_gated_boundary"   # the NEW best — drawn bold black
BASELINE = "train_run_esm2"            # also drawn bold (grey) to show total progress made
# Draw winner last/on top; keep a stable order for the rest.
ORDER = list(LABELS.keys())

# Distinct qualitative colors (tab20-ish), explicitly avoiding two similar blues.
PALETTE = {
    "train_run_esm2": "#404040",                       # baseline (bold dashed, set in plot)
    "esm2_telescoping_segmental": "#ff7f0e",           # orange
    "esm2_aho_mid_fusion_raw_m64": "#2ca02c",          # green
    "esm2_aho_emission_fusion": "#d62728",             # red
    "esm2_aho_emission_fusion_h32": "#9467bd",         # purple
    "train_run_esm2+3di_proj": "#8c564b",              # brown
    "train_run_esm2+3di_proj_gated_conv": "#e377c2",   # pink
    "train_run_esm2_aft_single_gated": "#bcbd22",      # olive
    "train_run_esmc_600m": "#17becf",                  # cyan
    "esmc6b_boundary_bond": "#1f77b4",                 # blue (old best)
}


# Fixed 5-aa bins (last one 45-50 to cover the cap at 50): averages neighbouring lengths
# so single-length noise on thin tails doesn't read as a scary cliff.
BIN_EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]   # [lo, hi) pairs; last bin 45..50
BIN_CENTERS = [(BIN_EDGES[i] + BIN_EDGES[i + 1] - 1) / 2 for i in range(len(BIN_EDGES) - 1)]
BIN_LABELS = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]-1}" for i in range(len(BIN_EDGES) - 1)]


def recall_binned(df, task):
    """Count-weighted recall in fixed 5-aa bins. Returns (centers, recall, n-per-bin)."""
    t = df[(df["kind"] == "true") & (df["task"] == task)]
    rec, npb = [], []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        sub = t[(t["length"] >= lo) & (t["length"] < hi)]
        npb.append(len(sub))
        rec.append(sub["matched"].mean() if len(sub) else np.nan)
    return np.array(BIN_CENTERS), np.array(rec), np.array(npb)


# task -> (output filename, title noun)
TASK_OUT = {
    "peptides": ("recall_by_length_models.png", "peptide"),
    "propeptides": ("recall_by_length_models_propeptides.png", "propeptide"),
}


def plot_task(task, dframes):
    ordered = [r for r in ORDER if r in dframes] + [r for r in dframes if r not in ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    print(f"=== {task} ===")
    for run in ordered:
        xs, rec, _ = recall_binned(dframes[run], task)
        is_win = run == WINNER
        is_base = run == BASELINE
        bold = is_win or is_base
        ax.plot(xs, rec, marker="o", markersize=5 if bold else 3,
                lw=3.2 if bold else 1.4,
                zorder=10 if is_win else 9 if is_base else 3,
                color="black" if is_win else ("#404040" if is_base else PALETTE.get(run, "#999999")),
                linestyle="--" if is_base else "-",
                alpha=1.0 if bold else 0.8,
                label=LABELS.get(run, run))
        print(f"{run:42s} " + " ".join(f"{lab}={v:.3f}" if not np.isnan(v) else f"{lab}=nan"
                                       for lab, v in zip(BIN_LABELS, rec)))
    fname, noun = TASK_OUT[task]
    ax.set_xticks(BIN_CENTERS); ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_xlim(LMIN - 0.5, LMAX + 0.5); ax.set_ylim(0, 1)
    ax.set_xlabel(f"true {noun} length (aa, 5-aa bins)"); ax.set_ylabel("recall (±3 matching)")
    ax.set_title(f"Recall by {noun} length, per model")
    ax.grid(alpha=.3); ax.legend(fontsize=7.5, ncol=2, loc="lower center", framealpha=.9)
    fig.tight_layout()
    out = FIG / fname
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}  ({len(ordered)} models)\n")


def main():
    csvs = sorted(STATS.glob("*__segments.csv"))
    dframes = {p.name[:-len("__segments.csv")]: pd.read_csv(p) for p in csvs}
    if not dframes:
        raise SystemExit(f"no *__segments.csv in {STATS}")
    for task in TASK_OUT:
        plot_task(task, dframes)


if __name__ == "__main__":
    main()
