#!/usr/bin/env python3
"""Soft (log-linear) ensemble of two CRF segment models, decode-time only (no retrain).

Each model contributes its final per-residue emissions [L, 101] (captured just before
crf.decode, so for the boundary model this already includes the boundary-state biases)
AND its learned transition matrix. We average BOTH across the two models and Viterbi-
decode the combined CRF (the allowed-transition structure is shared, so this is valid).
This moves along/outside the precision-recall frontier: if the two models have
DECORRELATED errors (different embeddings), the union of what they each get right can
push both P and R up — the "take the best of both profiles" the analysis pointed at.

The constituent models must score the SAME test proteins in the SAME order (same
data_file + partitioning, shuffle=False) so emissions align by protein name.

Self-check: ensembling a model WITH ITSELF must reproduce its solo metrics exactly.

Run from repo root:
  env/bin/python analysis/errors/src/ensemble_marginal.py --pairs A:B C:D [--device 0]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(ROOT))
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from analysis.errors.src.error_analysis import match_protein
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)


def prf(tp, fn, fp):
    r = tp / (tp + fn) if (tp + fn) else 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_model_and_emissions(run, device):
    """Return (model, {name: emissions[L,101] cpu tensor}, names, data, transitions)."""
    run_dir = ROOT / "runs" / run
    args = load_run_args(run_dir, SimpleNamespace(batch_size=None, device=None))
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    _, _, test_loader = get_dataloaders(args, device=device)
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

    captured = []
    orig_decode = model.crf.decode
    def hook(emissions=None, mask=None, **kw):
        m = mask.bool()
        for b in range(emissions.shape[0]):
            L = int(m[b].sum().item())
            captured.append(emissions[b, :L].detach().float().cpu())
        return orig_decode(emissions=emissions, mask=mask, **kw)
    model.crf.decode = hook
    run_dataloader(test_loader, model, optimizer=None, do_train=False, device=device,
                   collect_outputs=True, desc=f"emit {run}")
    model.crf.decode = orig_decode  # restore for later decoding

    names = list(test_loader.dataset.names)
    assert len(captured) == len(names), (len(captured), len(names))
    emis = {names[i]: captured[i] for i in range(len(names))}
    data = test_loader.dataset.data
    # Offload the (big) feature extractor off-GPU: we only need this model's tiny CRF
    # for decoding later. Prevents OOM when a second large model is loaded for a pair.
    model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    trans = model.crf.transitions.detach().float().cpu().clone()
    return model, emis, names, data, trans


def decode_emissions(crf, E, device):
    """Viterbi-decode a single protein's [L,101] emissions; return the state path."""
    L = E.shape[0]
    em = E.unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)
    out = crf.decode(emissions=em, mask=mask, top_k=1)
    paths = out[0] if isinstance(out, tuple) else out
    p = paths[0]
    if isinstance(p, list) and p and isinstance(p[0], list):  # top_k nesting
        p = p[0]
    return list(p)


def score_paths(paths_by_name, data):
    typed = {"tp": 0, "fn": 0, "fp": 0}
    anyc = {"tp": 0, "fn": 0, "fp": 0}
    for name, path in paths_by_name.items():
        row = data.loc[name]
        tpep = [(int(a), int(b)) for a, b in row["true_peptides"]]
        tpro = [(int(a), int(b)) for a, b in row["true_propeptides"]]
        ppep = [(int(a), int(b)) for a, b in convert_path_to_peptide_borders(path, PEPTIDE_START_STATE, PEPTIDE_END_STATE, 1)]
        ppro = [(int(a), int(b)) for a, b in convert_path_to_peptide_borders(path, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE, 1)]
        for tt, pp in [(tpep, ppep), (tpro, ppro)]:
            g, pm = match_protein(tt, pp, tol=3)
            for gr in g:
                typed["tp" if gr["matched"] else "fn"] += 1
            typed["fp"] += sum(1 for m in pm if not m)
        g, pm = match_protein(tpep + tpro, ppep + ppro, tol=3)
        for gr in g:
            anyc["tp" if gr["matched"] else "fn"] += 1
        anyc["fp"] += sum(1 for m in pm if not m)
    return prf(**typed), prf(**anyc)


def ensemble_pair(runA, runB, device, wA=0.5):
    mA, eA, names, data, tA = load_model_and_emissions(runA, device)
    if runB == runA:
        mB, eB, tB = mA, eA, tA
    else:
        mB, eB, _, _, tB = load_model_and_emissions(runB, device)
    common = [n for n in names if n in eB]
    crf = mA.crf.to(device)
    orig = crf.transitions.data.clone()
    crf.transitions.data = (wA * tA + (1 - wA) * tB).to(device)
    paths = {}
    with torch.no_grad():
        for n in common:
            E = wA * eA[n] + (1 - wA) * eB[n]
            paths[n] = decode_emissions(crf, E, device)
    crf.transitions.data = orig
    return score_paths(paths, data), len(common)


def solo(run, device):
    m, e, names, data, t = load_model_and_emissions(run, device)
    crf = m.crf.to(device)
    paths = {}
    with torch.no_grad():
        for n in names:
            paths[n] = decode_emissions(crf, e[n], device)
    return score_paths(paths, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B pairs to ensemble")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--selfcheck", type=str, default="esmc6b_boundary_bond")
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    bt = {r["run"]: r for r in csv.DictReader(open(ROOT / "analysis/metrics/big_metrics_table.csv"))}
    def ref(run):
        r = bt.get(run, {})
        return tuple(float(r.get(k)) if r.get(k) not in (None, "", "N/A") else None
                     for k in ("new_p_all", "new_r_all", "new_f1_all"))

    print(f"=== self-check: ensemble({args.selfcheck} WITH ITSELF) must match solo ===")
    (st, sa), n = ensemble_pair(args.selfcheck, args.selfcheck, device)
    rp, rr, rf = ref(args.selfcheck)
    print(f"  self-ens typed P/R/F1 = {st[0]:.4f}/{st[1]:.4f}/{st[2]:.4f}  | big_table = {rp}/{rr}/{rf}")
    print(f"  |ΔF1| vs big_table = {abs(st[2]-rf):.4f} ({'OK' if abs(st[2]-rf)<0.003 else 'CHECK pipeline'})\n")

    print(f"{'pair':55s} {'P_all':>7} {'R_all':>7} {'F1_all':>7} {'F1_any':>7}")
    for run in {p for pair in args.pairs for p in pair.split(':')}:
        (st, sa) = solo(run, device)
        print(f"{('SOLO '+run):55s} {st[0]:7.3f} {st[1]:7.3f} {st[2]:7.3f} {sa[2]:7.3f}")
    for pair in args.pairs:
        a, b = pair.split(":")
        (st, sa), n = ensemble_pair(a, b, device)
        print(f"{('ENS  '+a+' + '+b):55s} {st[0]:7.3f} {st[1]:7.3f} {st[2]:7.3f} {sa[2]:7.3f}   (n={n})")


if __name__ == "__main__":
    main()
