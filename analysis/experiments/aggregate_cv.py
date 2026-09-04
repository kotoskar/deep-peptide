"""Aggregate the 20 cells of a 5x4 nested-CV model into runs/<name>/nested_cv_summary.json.

Same aggregation as nested_cv_driver.py: an outer fold's score is the mean over its
four inner cells, the CV estimate is the mean over the five outer folds, and the
reported spread is the std across outer folds (the four cells of one outer fold share
a test set and are not independent).

Usage: python3 analysis/experiments/aggregate_cv.py <model name> [--n_folds 5] [--partial]
  --partial  write a summary even if fewer than 20 cells are done (marked incomplete).
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()

    pairs = [(o, k) for o in range(a.n_folds) for k in range(a.n_folds) if k != o]
    root = Path("runs") / a.name
    cells = []
    for o, k in pairs:
        p = root / f"outer{o}_inner{k}" / "cell_result.json"
        if p.exists():
            cells.append(json.load(open(p)))
    print(f"{a.name}: {len(cells)}/{len(pairs)} cells done")
    if not cells or (len(cells) < len(pairs) and not a.partial):
        print("not complete; pass --partial to summarise what is there")
        return 1

    df = pd.DataFrame(cells)
    per_outer = df.groupby("outer").mean(numeric_only=True)
    summary = {
        "name": a.name,
        "n_cells": len(df),
        "complete": len(df) == len(pairs),
        "mean_epoch": float(df["best_epoch"].mean()),
        "std_epoch": float(df["best_epoch"].std()),
        "per_outer_f1_all": per_outer["test_f1_all"].round(4).to_dict(),
        "cv_f1_all_mean": float(per_outer["test_f1_all"].mean()),
        "cv_f1_all_std": float(per_outer["test_f1_all"].std()),
        "cv_f1_peptides_mean": float(per_outer["test_f1_peptides"].mean()),
        "cv_f1_peptides_std": float(per_outer["test_f1_peptides"].std()),
        "cv_f1_propeptides_mean": float(per_outer["test_f1_propeptides"].mean()),
        "cv_f1_propeptides_std": float(per_outer["test_f1_propeptides"].std()),
        "cv_precision_all_mean": float(per_outer["test_precision_all"].mean()),
        "cv_recall_all_mean": float(per_outer["test_recall_all"].mean()),
        "note": "F1 here is the training-time (original, uncorrected) matcher at +-3; "
                "the corrected-matcher rescoring is done at home from each cell's model.pt.",
    }
    json.dump(summary, open(root / "nested_cv_summary.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("cv_f1_all_mean", "cv_f1_all_std", "mean_epoch")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
