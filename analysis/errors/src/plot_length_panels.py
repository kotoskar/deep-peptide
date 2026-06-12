#!/usr/bin/env python3
"""3×2 length panels: (counts / recall / precision) × (peptides / propeptides).

Recall alone is misleading on the long tail (the new 3Di model predicts fewer but more
precise long peptides → recall down, precision up, F1 maintained). This figure puts
recall and precision side by side on the SAME 5-aa length bins and aligned axes so the
trade reads at a glance, with a count row on top for the sample-size context.

Conventions:
  - counts: number of TRUE segments per length bin (model-independent support).
  - recall@bin = matched true segments / all true segments (binned by TRUE length).
  - precision@bin = matched predictions / all predictions (binned by PREDICTED length;
    needs the tp_pred rows error_analysis now emits).

Reads every analysis/errors/error_stats/<run>__segments.csv. Run from repo root:
  env/bin/python analysis/errors/src/plot_length_panels.py
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

BIN_EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]
CENTERS = [(BIN_EDGES[i] + BIN_EDGES[i + 1] - 1) / 2 for i in range(len(BIN_EDGES) - 1)]
LABELS_X = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]-1}" for i in range(len(BIN_EDGES) - 1)]

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
WINNER, BASELINE = "esmc6b_3di_gated_boundary", "train_run_esm2"
PALETTE = {
    "train_run_esm2": "#404040",
    "esm2_telescoping_segmental": "#ff7f0e",
    "esm2_aho_mid_fusion_raw_m64": "#2ca02c",
    "esm2_aho_emission_fusion": "#d62728",
    "esm2_aho_emission_fusion_h32": "#9467bd",
    "train_run_esm2+3di_proj": "#8c564b",
    "train_run_esm2+3di_proj_gated_conv": "#e377c2",
    "train_run_esm2_aft_single_gated": "#bcbd22",
    "train_run_esmc_600m": "#17becf",
    "esmc6b_boundary_bond": "#1f77b4",
}
ORDER = list(LABELS.keys())


def binstat(df, task, kind_true, value):
    """value='recall' over kind=='true'; value='precision' over predicted (tp/fp)."""
    out = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        if value == "recall":
            s = df[(df.kind == "true") & (df.task == task) & (df.length >= lo) & (df.length < hi)]
            out.append(s.matched.mean() if len(s) else np.nan)
        else:  # precision over predicted segments
            s = df[(df.kind.isin(["tp_pred", "fp_pred"])) & (df.task == task)
                   & (df.length >= lo) & (df.length < hi)]
            out.append(s.matched.mean() if len(s) else np.nan)
    return np.array(out)


def true_counts(df, task):
    t = df[(df.kind == "true") & (df.task == task)]
    return np.array([((t.length >= BIN_EDGES[i]) & (t.length < BIN_EDGES[i + 1])).sum()
                     for i in range(len(BIN_EDGES) - 1)])


def style(run):
    is_w, is_b = run == WINNER, run == BASELINE
    return dict(lw=3.2 if (is_w or is_b) else 1.4,
                color="black" if is_w else ("#404040" if is_b else PALETTE.get(run, "#999")),
                linestyle="--" if is_b else "-",
                zorder=10 if is_w else 9 if is_b else 3,
                alpha=1.0 if (is_w or is_b) else 0.8,
                marker="o", markersize=5 if (is_w or is_b) else 3)


def _f1_labels():
    """run -> ' (F1 0.69)' suffix from the big table (new±3 all, fallback old)."""
    import csv
    bt = ROOT / "analysis" / "metrics" / "big_metrics_table.csv"
    out = {}
    if bt.exists():
        for r in csv.DictReader(open(bt)):
            v = r.get("new_f1_all") or ""
            if v in ("", "N/A"):
                v = r.get("old_f1_all") or ""
            try:
                out[r["run"]] = f" (F1 {float(v):.2f})"
            except ValueError:
                out[r["run"]] = ""
    return out


def main():
    dframes = {p.name[:-len("__segments.csv")]: pd.read_csv(p)
               for p in sorted(STATS.glob("*__segments.csv"))}
    ordered = [r for r in ORDER if r in dframes] + [r for r in dframes if r not in ORDER]
    has_pred = any("tp_pred" in d.kind.values for d in dframes.values())
    f1lab = _f1_labels()

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    x = np.array(CENTERS)
    for col, task in enumerate(["peptides", "propeptides"]):
        # row 0: counts (one curve per class — use baseline df, true counts are model-independent)
        ax = axes[0][col]
        c = true_counts(dframes[BASELINE], task)
        col_c = "#2b6cb0" if task == "peptides" else "#c05621"
        ax.fill_between(x, c, color=col_c, alpha=0.35)
        ax.plot(x, c, color=col_c, lw=2)
        ax.set_title(f"{task}", fontsize=12, fontweight="bold")
        ax.set_ylabel("# true segments" if col == 0 else "")
        # row 1: recall, row 2: precision
        for ri, value in [(1, "recall"), (2, "precision")]:
            ax = axes[ri][col]
            for run in ordered:
                y = binstat(dframes[run], task, "true", value)
                lab = LABELS.get(run, run) + (f1lab.get(run, "") if (ri == 1 and col == 0) else "")
                ax.plot(x, y, label=lab, **style(run))
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(f"{value} (±3)")
    for ax in axes[2]:
        ax.set_xticks(CENTERS); ax.set_xticklabels(LABELS_X, fontsize=8, rotation=0)
        ax.set_xlabel("true / predicted length (aa, 5-aa bins)")
    for row in axes:
        for ax in row:
            ax.grid(alpha=.3)
    # one shared legend below
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, framealpha=.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Length panels — counts / recall / precision × peptides / propeptides", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    out = FIG / "length_panels.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  | precision available: {has_pred}")
    # console: winner vs baseline vs old-winner recall&precision on long peptides
    for run in [BASELINE, "esmc6b_boundary_bond", WINNER]:
        r = binstat(dframes[run], "peptides", "true", "recall")
        p = binstat(dframes[run], "peptides", "true", "precision")
        print(f"{run:34s} pep 35-39: rec={r[6]:.3f} prec={p[6]:.3f} | 40-44: rec={r[7]:.3f} prec={p[7]:.3f}")


if __name__ == "__main__":
    main()
