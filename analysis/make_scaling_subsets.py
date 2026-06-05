#!/usr/bin/env python3
"""Build CORRECTED data-scaling train subsets.

The original scaling runs (train_run_esm2_{25,50,75}) are unreliable: their
subset CSVs were made by `select_subdata_ids.py`, which subsampled EVERY graphpart
cluster — including valid (3) and test (4) — so smaller runs were evaluated on
SMALLER, different test sets (not comparable), and train_run_esm2_25 actually
points at labeled_sequences50.csv (mislabeled).

This rebuilds a clean series: subsample TRAIN clusters (0,1,2) only, keep
valid(3)+test(4) FULL and identical across fractions, seeded. 100% = the existing
clean run train_run_esm2_100 (full labeled_sequences.csv).

Writes data/uniprot_2022/scaling/labeled_sequences_trainfrac{50,60,70,80,90}.csv
"""
from pathlib import Path
import pandas as pd

SEED = 42
FRACS = [50, 60, 70, 80, 90]
OUT = Path("data/uniprot_2022/scaling"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv("data/uniprot_2022/labeled_sequences.csv")
    gp = pd.read_csv("data/uniprot_2022/graphpart_assignments.csv")
    ac2cl = dict(zip(gp["AC"], gp["cluster"].astype(float).astype(int)))
    df["_cluster"] = df["protein_id"].map(ac2cl)
    train_full = df[df["_cluster"].isin([0, 1, 2])]
    heldout = df[df["_cluster"].isin([3, 4])]  # valid+test, always full
    for frac in FRACS:
        sub = train_full.groupby("_cluster", group_keys=False).apply(
            lambda g: g.sample(frac=frac / 100, random_state=SEED))
        out = pd.concat([sub, heldout]).drop(columns="_cluster")
        path = OUT / f"labeled_sequences_trainfrac{frac}.csv"
        out.to_csv(path, index=False)
        print(f"frac{frac}: train={len(sub)} + heldout={len(heldout)} = {len(out)} -> {path}")


if __name__ == "__main__":
    main()
