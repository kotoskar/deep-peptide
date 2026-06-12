#!/usr/bin/env python3
"""Evaluate every run on the VALIDATION partition (cluster 3) and write
valid_metrics_infer.json per run + a valid summary CSV.

Rationale: model selection should methodologically be judged on validation, not test.
We selected/compared on test; this lets us confirm the val ranking REPRODUCES the test
ranking. If it diverges, we overfit to test and conclusions need revisiting.

Reuses infer.py's exact fp32 eval (load_run_args, get_dataloaders, get_model,
load_state_dict, evaluate_loader) — same determinism/seed/fp32 as the test pipeline,
just on the valid loader (the middle element of get_dataloaders).

Run from repo root:
  env/bin/python analysis/metrics/src/infer_valid.py [--only r1 r2 ...] [--device 0]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(ROOT))
from infer import load_run_args, load_state_dict, evaluate_loader, to_jsonable
from src.train_loop_crf import get_dataloaders, get_model

SKIP = {"aho_only", "aho_only_fixed", "joined_error_analysis_test"}


def eval_run(run_dir: Path, device: str) -> dict:
    args = load_run_args(run_dir, SimpleNamespace(batch_size=None, device=None, seed=42))
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    # middle element = validation loader (cluster 3 by default)
    _, valid_loader, _ = get_dataloaders(args, device=device)
    model = get_model(args).to(device)
    fe = getattr(model, "feature_extractor", None)
    bilstm = getattr(fe, "biLSTM", None) if fe is not None else None
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
    _, metrics, _ = evaluate_loader(valid_loader, model, device, args)
    with (run_dir / "valid_metrics_infer.json").open("w") as f:
        json.dump(to_jsonable(metrics), f, indent=2, ensure_ascii=False)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    runs = args.only or sorted(d.name for d in (ROOT / "runs").iterdir()
                               if (d / "model.pt").exists() and (d / "config.json").exists()
                               and d.name not in SKIP)
    rows = []
    for name in runs:
        rd = ROOT / "runs" / name
        try:
            m = eval_run(rd, device)
            rows.append({"run": name, **{k: m.get(k) for k in (
                "f1 all", "precision all", "recall all",
                "residue mcc all", "residue roc_auc all",
                "f1 peptides", "f1 propeptides")}})
            print(f"[ok] {name:42s} val F1_all={m.get('f1 all'):.4f} MCC={m.get('residue mcc all'):.4f}")
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    out = ROOT / "analysis" / "metrics" / "valid_metrics_infer_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "f1 all", "precision all", "recall all",
                                          "residue mcc all", "residue roc_auc all",
                                          "f1 peptides", "f1 propeptides"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
