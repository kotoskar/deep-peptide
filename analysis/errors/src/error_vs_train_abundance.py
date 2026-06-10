#!/usr/bin/env python3
"""Per-organism test error vs how many TRAIN examples that organism has.

Develops the "more data" narrative: the model fails on organisms that are
under-represented in the training set. For each organism we plot the number of
TRAIN peptide segments (clusters 0,1,2) against the model's TEST peptide recall
(baseline ESM2), i.e. error concentrates where training coverage is thin.

Outputs: texs/error_analysis/figures/error_vs_train_abundance.png + a CSV.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())))
from src.utils.crf_label_utils import parse_coordinate_string

DATA = "data/uniprot_2022/labeled_sequences.csv"
GP = "data/uniprot_2022/graphpart_assignments.csv"
SEG = "analysis/errors/error_stats/all_segments.csv"
MODEL = "train_run_esm2"          # baseline; the canonical reference model
FIG = Path("texs/error_analysis/figures")
OUTCSV = Path("analysis/errors/error_stats/error_vs_train_abundance.csv")
TRAIN_CLUSTERS = {0, 1, 2}


def species(o):
    return o.split("(")[0].strip() if isinstance(o, str) else "unknown"


def train_peptide_counts():
    df = pd.read_csv(DATA, index_col=0)
    gp = pd.read_csv(GP, index_col="AC")
    cl = {ac: int(float(c)) for ac, c in gp["cluster"].items()}
    counts = {}
    for _, row in df.iterrows():
        ac = row["protein_id"]
        if cl.get(ac) not in TRAIN_CLUSTERS:
            continue
        org = species(row.get("organism"))
        if pd.isna(row["coordinates"]):
            continue
        n = len(parse_coordinate_string(str(row["coordinates"]), merge_overlaps=True))
        counts[org] = counts.get(org, 0) + n
    return counts  # organism -> # train peptide segments


def main():
    train_counts = train_peptide_counts()
    seg = pd.read_csv(SEG)
    seg = seg[(seg["run"] == MODEL) & (seg["task"] == "peptides") & (seg["kind"] == "true")]

    rows = []
    for org, sub in seg.groupby("organism"):
        rows.append({
            "organism": org,
            "train_peptides": train_counts.get(org, 0),
            "test_peptides": len(sub),
            "test_recall": sub["matched"].mean(),
            "test_errors": int((~sub["matched"]).sum()),
        })
    d = pd.DataFrame(rows).sort_values("train_peptides")
    d.to_csv(OUTCSV, index=False)

    # focus on organisms with a meaningful test count
    m = d[d["test_peptides"] >= 15].copy()
    # spearman (rank) correlation is robust to the heavy-tailed counts
    from scipy.stats import spearmanr, pearsonr
    x = m["train_peptides"].to_numpy(float)
    y = m["test_recall"].to_numpy(float)
    sp = spearmanr(x, y).correlation
    pe = pearsonr(np.log10(x + 1), y)[0]

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x + 1, y, s=np.clip(m["test_peptides"], 15, 400) / 4, alpha=0.7, color="#3b78c2")
    for _, r in m.iterrows():
        ax.annotate(r["organism"].split()[0], (r["train_peptides"] + 1, r["test_recall"]),
                    fontsize=6.5, alpha=0.75, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("# peptide segments of this organism in TRAIN (log scale)")
    ax.set_ylabel("TEST peptide recall (±3, baseline ESM2)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Errors concentrate on organisms thin in train\n"
                 f"Spearman ρ={sp:.2f} (recall vs train count); marker size ∝ #test peptides")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_train_abundance.png", dpi=130)
    plt.close(fig)

    print(f"Spearman ρ(train_count, recall) = {sp:.3f}; Pearson(log train, recall) = {pe:.3f}")
    print(m[["organism", "train_peptides", "test_peptides", "test_recall"]].to_string(index=False))
    print(f"\nwrote {FIG}/error_vs_train_abundance.png and {OUTCSV}")


if __name__ == "__main__":
    main()
