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
BINS = {"5": (5, 5), "6-10": (6, 10), "11-20": (11, 20), "21-30": (21, 30), "31-50": (31, 50)}

# Short display labels; runs without an entry fall back to the folder name.
LABELS = {
    "train_run_esm2": "ESM2 (baseline)",
    "esm2_telescoping_segmental": "ESM2 telescoping",
    "esm2_aho_mid_fusion_raw_m64": "ESM2 Aho mid-fusion",
    "esm2_aho_emission_fusion": "ESM2 Aho emis-fusion",
    "esm2_aho_emission_fusion_h32": "ESM2 Aho emis-fusion h32",
    "train_run_esm2+3di_proj": "ESM2+3Di proj",
    "train_run_esm2+3di_proj_gated_conv": "ESM2+3Di proj gated conv",
    "train_run_esm2_aft_single_gated": "ESM2 AFT single gated",
    "train_run_esmc_600m": "ESM-C 600M",
    "esmc6b_boundary_bond": "ESM-C 6B + boundary/bond ★",
}
# Draw winner (if present) bold/on top; keep a stable order for the rest.
ORDER = list(LABELS.keys())


def lenbin(L):
    for lab, (lo, hi) in BINS.items():
        if lo <= L <= hi:
            return lab
    return None


def recall_by_bin(df, task):
    t = df[(df["kind"] == "true") & (df["task"] == task)].copy()
    t["lenbin"] = t["length"].apply(lenbin)
    return [t[t["lenbin"] == lab]["matched"].mean() if len(t[t["lenbin"] == lab]) else np.nan
            for lab in LEN_LABELS]


# task -> (output filename, title noun)
TASK_OUT = {
    "peptides": ("recall_by_length_models.png", "peptide"),
    "propeptides": ("recall_by_length_models_propeptides.png", "propeptide"),
}


def plot_task(task, dframes):
    runs = {run: recall_by_bin(df, task) for run, df in dframes.items()}
    ordered = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]
    x = np.arange(len(LEN_LABELS))
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    print(f"=== {task} ===")
    for i, run in enumerate(ordered):
        winner = run == "esmc6b_boundary_bond"
        ax.plot(x, runs[run], marker="o",
                lw=3.0 if winner else 1.6, zorder=5 if winner else 2,
                color="black" if winner else cmap(i % 10),
                label=LABELS.get(run, run))
        print(f"{run:42s} " + "  ".join(f"{lab}={v:.3f}" if not np.isnan(v) else f"{lab}=nan"
                                        for lab, v in zip(LEN_LABELS, runs[run])))
    fname, noun = TASK_OUT[task]
    ax.set_xticks(x); ax.set_xticklabels(LEN_LABELS)
    ax.set_xlabel(f"true {noun} length (aa)"); ax.set_ylabel("recall (±3 matching)")
    ax.set_title(f"Recall by {noun} length, per model"); ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, ncol=2, loc="lower center")
    fig.tight_layout()
    out = FIG / fname
    fig.savefig(out, dpi=130); plt.close(fig)
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
