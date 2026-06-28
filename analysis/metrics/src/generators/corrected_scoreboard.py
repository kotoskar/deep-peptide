"""Recompute the clean-split MODEL-SELECT fold{2} scoreboard with BOTH matchers
(orig=bug get_counts_for_protein, corr=corrected match_protein) for every 2026_* model.
Each model's config points at its own covered-split partitioning_file; we eval test_partitions=[2].
Writes analysis/metrics/clean_split_modelselect.csv. GPU inference only.
Run from repo root, PYTHONPATH=.
"""
import os, sys, glob, csv
sys.path.insert(0, os.getcwd())
from pathlib import Path
from types import SimpleNamespace
import torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders, get_counts_for_protein,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def prf(tp, fn, fp):
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return (2*p*r/(p+r) if p+r else 0.0)

def corr_counts(true, pred, tol=3):
    g, pm = match_protein(true, pred, tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)

runs = sorted(d for d in glob.glob("runs/2026_*") if os.path.exists(os.path.join(d,"model.pt")))
rows = []
for run_dir in runs:
    name = os.path.basename(run_dir)
    try:
        args = load_run_args(Path(run_dir), SimpleNamespace(batch_size=None, device=None, seed=42))
        _, _, test_loader = get_dataloaders(args, train_partitions=[0,3,4,6], valid_partitions=[1],
                                            test_partitions=[2], device=device)
        model = get_model(args).to(device)
        bil = getattr(getattr(model,"feature_extractor",None),"biLSTM",None)
        if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
        model.load_state_dict(load_state_dict(os.path.join(run_dir,"model.pt"), device)); model.eval()
        _,_,preds,_,_ = run_dataloader(test_loader, model, optimizer=None, do_train=False,
                                       device=device, collect_outputs=True, desc=name)
        data = test_loader.dataset.data; names = test_loader.dataset.names
        acc = {"orig":{"pep":[0,0,0],"pp":[0,0,0]}, "corr":{"pep":[0,0,0],"pp":[0,0,0]}}
        for i,pred in enumerate(preds):
            row = data.loc[names[i]]
            for task,(ss,es),key in [("peptides",(PEPTIDE_START_STATE,PEPTIDE_END_STATE),"pep"),
                                     ("propeptides",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE),"pp")]:
                true=[(int(a),int(b)) for a,b in (row["true_peptides"] if task=="peptides" else row["true_propeptides"])]
                pb=convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)
                o=get_counts_for_protein(true,pb,3) if (true or pb) else (0,0,0)
                c=corr_counts(true,pb,3)
                for k in range(3): acc["orig"][key][k]+=o[k]; acc["corr"][key][k]+=c[k]
        out={"run":name,"n_test":len(names)}
        for kind in ("orig","corr"):
            pep=acc[kind]["pep"]; pp=acc[kind]["pp"]; allc=[pep[k]+pp[k] for k in range(3)]
            out[f"{kind}_all"]=round(prf(*allc),4); out[f"{kind}_pep"]=round(prf(*pep),4); out[f"{kind}_propep"]=round(prf(*pp),4)
        rows.append(out)
        print(f"[OK] {name:32s} bug_all={out['orig_all']:.4f} corr_all={out['corr_all']:.4f} (Δ{out['corr_all']-out['orig_all']:+.3f})", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)

rows.sort(key=lambda r: -r["corr_all"])
cols=["run","n_test","corr_all","corr_pep","corr_propep","orig_all","orig_pep","orig_propep"]
with open("analysis/metrics/clean_split_modelselect.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k:r[k] for k in cols})
print("\n=== CORRECTED-matcher scoreboard (model-select fold{2}, sorted) ===")
for r in rows: print(f"  {r['run'][5:]:30s} corr_all={r['corr_all']:.4f} pep={r['corr_pep']:.4f} propep={r['corr_propep']:.4f}  (bug_all={r['orig_all']:.4f})")
print(f"\nwrote analysis/metrics/clean_split_modelselect.csv ({len(rows)} models)")
