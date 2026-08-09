#!/usr/bin/env python3
"""Corrected (dual-reporting) ±3 peptide/propeptide metric for nested-CV cells.

Same bug as `corrected_metrics.py` (see that file's docstring and
`src/utils/manuscript_metrics.py::get_counts_for_protein`), but nested-CV
cells (runs/5cv_*/outer{o}_inner{k}/) each hold out a DIFFERENT test
partition, recorded per-cell in cell_result.json (train/valid/test_partitions),
not the fixed global test split. error_analysis.run_inference() cannot be
reused as-is because it calls get_dataloaders() with the hardcoded default
partitions -- this script passes the cell's own partitions through instead.

For every cell: run TEST-fold inference, decode peptide/propeptide borders,
compute P/R/F1 (peptides/propeptides/all, tol=3) both ways (orig=published
buggy matcher, corr=fixed matcher). Write per-cell
runs/<name>/outer{o}_inner{k}/corrected_metrics.json, then aggregate across
the 5 outer folds (mean of the 4 inner cells per outer, then mean+std across
outers -- same procedure as nested_cv_driver.aggregate) into
runs/<name>/nested_cv_summary_corrected.json.

Usage: env/bin/python analysis/metrics/src/corrected_metrics_cv.py [--models NAME ...] [--device 0]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())))
import pandas as pd
import torch

from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders, get_counts_for_protein,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)
from analysis.errors.src.error_analysis import match_protein

N_FOLDS = 5
DEFAULT_MODELS = ["5cv_baseline_esm2", "5cv_esm2_adapter_only", "5cv_esm2_boundary",
                  "5cv_esm2_full", "5cv_esmc6b_plain"]
STATES = {"peptides": (PEPTIDE_START_STATE, PEPTIDE_END_STATE),
          "propeptides": (PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)}


def corrected_counts(true, pred, tol=3):
    g, pm = match_protein(true, pred, tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)


def prf(tp, fn, fp):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def cell_inference(run_dir: Path, device: str, train_partitions, valid_partitions, test_partitions):
    args = load_run_args(run_dir, SimpleNamespace(batch_size=None, device=None))
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    _, _, test_loader = get_dataloaders(args, train_partitions=train_partitions,
                                        valid_partitions=valid_partitions,
                                        test_partitions=test_partitions, device=device)
    model = get_model(args).to(device)
    if getattr(args, "feature_extractor", None) == "LSTMCNN" and hasattr(model, "feature_extractor"):
        bilstm = getattr(model.feature_extractor, "biLSTM", None)
        if bilstm is not None and hasattr(bilstm, "flatten_parameters"):
            bilstm.flatten_parameters()
    sd = load_state_dict(run_dir / "model.pt", device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        buf = set(dict(model.named_buffers()).keys())
        inc = model.load_state_dict(sd, strict=False)
        if not set(inc.missing_keys).issubset(buf) or inc.unexpected_keys:
            raise
    model.eval()
    _, _, preds, _, _ = run_dataloader(test_loader, model, optimizer=None, do_train=False,
                                       device=device, collect_outputs=True,
                                       desc=f"infer {run_dir.name} [test]")
    data = test_loader.dataset.data
    names = test_loader.dataset.names
    return preds, names, data


def score_cell(run_dir: Path, device: str, cell: dict) -> dict:
    preds, names, data = cell_inference(run_dir, device, cell["train_partitions"],
                                        cell["valid_partitions"], cell["test_partitions"])
    orig = {"peptides": [0, 0, 0], "propeptides": [0, 0, 0]}
    corr = {"peptides": [0, 0, 0], "propeptides": [0, 0, 0]}
    for i, pred in enumerate(preds):
        row = data.loc[names[i]]
        for task, (ss, es) in STATES.items():
            true = [(int(a), int(b)) for a, b in
                    (row["true_peptides"] if task == "peptides" else row["true_propeptides"])]
            pb = convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)
            otp, ofn, ofp = get_counts_for_protein(true, pb, 3) if (true or pb) else (0, 0, 0)
            ctp, cfn, cfp = corrected_counts(true, pb, 3)
            orig[task][0] += otp; orig[task][1] += ofn; orig[task][2] += ofp
            corr[task][0] += ctp; corr[task][1] += cfn; corr[task][2] += cfp
    out = {"outer": cell["outer"], "inner": cell["inner"]}
    for kind, acc in (("orig", orig), ("corr", corr)):
        allc = [acc["peptides"][k] + acc["propeptides"][k] for k in range(3)]
        for task, c in (("peptides", acc["peptides"]), ("propeptides", acc["propeptides"]), ("all", allc)):
            p, r, f = prf(*c)
            out[f"{kind}_{task}_precision"] = round(p, 6)
            out[f"{kind}_{task}_recall"] = round(r, 6)
            out[f"{kind}_{task}_f1"] = round(f, 6)
    return out


def aggregate(name: str, cells: list[dict]) -> dict:
    df = pd.DataFrame(cells)
    per_outer = df.groupby("outer").mean(numeric_only=True)
    summary = {"name": name, "n_cells": len(df)}
    for kind in ("orig", "corr"):
        for task in ("peptides", "propeptides", "all"):
            col = f"{kind}_{task}_f1"
            summary[f"cv_{kind}_{task}_f1_mean"] = float(per_outer[col].mean())
            summary[f"cv_{kind}_{task}_f1_std"] = float(per_outer[col].std())
        summary[f"per_outer_{kind}_all_f1"] = per_outer[f"{kind}_all_f1"].round(4).to_dict()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    for name in args.models:
        run_root = Path("runs") / name
        cells_meta = []
        for o in range(N_FOLDS):
            for k in range(N_FOLDS):
                if k == o:
                    continue
                cell_dir = run_root / f"outer{o}_inner{k}"
                cr = cell_dir / "cell_result.json"
                if not cr.exists():
                    print(f"[skip] {name} outer{o}_inner{k}: no cell_result.json")
                    continue
                cells_meta.append((cell_dir, json.load(open(cr))))

        results = []
        for cell_dir, cell in cells_meta:
            try:
                r = score_cell(cell_dir, device, cell)
            except Exception as e:
                print(f"[FAIL] {cell_dir}: {type(e).__name__}: {e}")
                continue
            json.dump(r, open(cell_dir / "corrected_metrics.json", "w"), indent=2)
            results.append(r)
            print(f"[OK] {cell_dir.name} f1_all orig={r['orig_all_f1']:.4f} corr={r['corr_all_f1']:.4f} "
                  f"(Drecall_all={r['corr_all_recall']-r['orig_all_recall']:+.4f})")

        if len(results) < len(cells_meta):
            print(f"[WARN] {name}: only {len(results)}/{len(cells_meta)} cells scored, skipping aggregate")
            continue
        summary = aggregate(name, results)
        out_path = run_root / "nested_cv_summary_corrected.json"
        json.dump(summary, open(out_path, "w"), indent=2)
        print(f"[model done] {name}: orig f1_all={summary['cv_orig_all_f1_mean']:.4f} "
              f"corr f1_all={summary['cv_corr_all_f1_mean']:.4f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
