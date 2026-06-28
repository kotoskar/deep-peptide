"""Master artifact for the thesis-by-thesis report figures. 9 models x folds {2},{5}.
For each model/fold/protein/task store, at tolerance tol in {0,1,2,3}:
  TRUE-segment rows: model,fold,protein,task,length,genus,new, m0,m1,m2,m3 (matched@tol)
  PRED-segment rows: model,fold,protein,task,plen, m0,m1,m2,m3 (this pred matched@tol)
From these: P/R/F1 at any tolerance, by any stratum, paired on the same proteins.
Also records per-model param count. Run: env/bin/python this.py  (from repo root)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from pathlib import Path
from types import SimpleNamespace
import pandas as pd, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders, PEPTIDE_START_STATE, PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MODELS = ["baseline_esm2","esm2_boundary_bond","esmc_600m","esmc_6b","esmc6b_boundary",
          "esmc6b_boundary_bond","esmc6b_3di_zeroctrl","esmc6b_3di_gated_boundary",
          "esmc6b_3di_nocompress"]
TOLS=[0,1,2,3]
lab2022 = set(pd.read_csv("data/uniprot_2022/labeled_sequences.csv")["protein_id"])

true_rows=[]; pred_rows=[]; params=[]
for run in MODELS:
    rd=f"runs/2026_{run}"
    args=load_run_args(Path(rd), SimpleNamespace(batch_size=None, device=None, seed=42))
    model=get_model(args).to(device)
    npar=sum(p.numel() for p in model.parameters())
    params.append((run,npar))
    bil=getattr(getattr(model,"feature_extractor",None),"biLSTM",None)
    if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
    model.load_state_dict(load_state_dict(os.path.join(rd,"model.pt"), device)); model.eval()
    for fold in (2,5):
        _,_,test=get_dataloaders(args, train_partitions=[0,3,4,6], valid_partitions=[1],
                                 test_partitions=[fold], device=device)
        _,_,preds,_,_=run_dataloader(test, model, optimizer=None, do_train=False,
                                     device=device, collect_outputs=True, desc=f"{run} f{fold}")
        data=test.dataset.data; names=test.dataset.names
        for i,pred in enumerate(preds):
            pid=names[i]; r=data.loc[pid]
            org=str(r.get("organism","")).split()[0] if pd.notna(r.get("organism")) else "?"
            new=pid not in lab2022
            for task,(ss,es) in [("pep",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),
                                 ("propep",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
                true=[(int(a),int(b)) for a,b in (r["true_peptides"] if task=="pep" else r["true_propeptides"])]
                pb=convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)
                # match at each tolerance
                gm={}; pm={}
                for tol in TOLS:
                    g,p=match_protein(true, pb, tol); gm[tol]=g; pm[tol]=p
                ng=len(gm[3])  # group count is tol-independent (grouping ignores tol)
                for gi in range(ng):
                    rec=dict(model=run,fold=fold,protein=pid,task=task,
                             length=gm[3][gi]["length"],genus=org,new=new)
                    for tol in TOLS: rec[f"m{tol}"]=int(bool(gm[tol][gi]["matched"]))
                    true_rows.append(rec)
                for pi,(psx,pex) in enumerate(pb):
                    rec=dict(model=run,fold=fold,protein=pid,task=task,plen=pex-psx+1)
                    for tol in TOLS: rec[f"m{tol}"]=int(bool(pm[tol][pi]))
                    pred_rows.append(rec)
    print(f"[done] {run}  params={npar:,}", flush=True)

pd.DataFrame(true_rows).to_csv("analysis/metrics/clean_tol_true.csv", index=False)
pd.DataFrame(pred_rows).to_csv("analysis/metrics/clean_tol_pred.csv", index=False)
pd.DataFrame(params,columns=["model","n_params"]).to_csv("analysis/metrics/clean_model_params.csv", index=False)
print(f"\nwrote clean_tol_true.csv ({len(true_rows)}), clean_tol_pred.csv ({len(pred_rows)}), clean_model_params.csv")
print("PARAMS:", {r:f"{n/1e6:.2f}M" for r,n in params})
