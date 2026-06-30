"""Per-segment (task-split) tp/fn/fp at ±3 for the ESM-2 runs missing from clean_tol_*,
needed by the unified effects slope figure. Pooled folds {2,5}, corrected match_protein.
Writes analysis/metrics/effects_esm2_extra.csv (model,fold,protein,task,tp,fn,fp). GPU.
Run from repo root: env/bin/python analysis/metrics/src/generators/effects_esm2_eval.py
"""
import os, sys; sys.path.insert(0, os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import pandas as pd, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")
TOL=3
MODELS=[("esm2_boundary","runs/2026_esm2_boundary"),
        ("esm2_3di_proj_gated_conv","runs/2026_esm2_3di_proj_gated_conv")]
def counts(true,pred,tol=TOL):
    g,pm=match_protein(true,pred,tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)
rows=[]
for name,run_dir in MODELS:
    args=load_run_args(Path(run_dir),SimpleNamespace(batch_size=None,device=None,seed=42))
    for fold in (2,5):
        _,_,test=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],
                                 test_partitions=[fold],device=device)
        model=get_model(args).to(device)
        bil=getattr(getattr(model,"feature_extractor",None),"biLSTM",None)
        if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
        model.load_state_dict(load_state_dict(os.path.join(run_dir,"model.pt"),device)); model.eval()
        _,_,preds,_,_=run_dataloader(test,model,optimizer=None,do_train=False,device=device,
                                     collect_outputs=True,desc=f"{name}:f{fold}")
        data=test.dataset.data; names=test.dataset.names
        for i,pred in enumerate(preds):
            row=data.loc[names[i]]
            for task,(ss,es),col in [("pep",(PEPTIDE_START_STATE,PEPTIDE_END_STATE),"true_peptides"),
                                     ("propep",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE),"true_propeptides")]:
                true=[(int(a),int(b)) for a,b in row[col]]
                pb=convert_path_to_peptide_borders(pred,start_state=ss,stop_state=es,offset=1)
                tp,fn,fp=counts(true,pb)
                rows.append({"model":name,"fold":fold,"protein":names[i],"task":task,"tp":tp,"fn":fn,"fp":fp})
        del model; torch.cuda.empty_cache()
    print(f"[done] {name}", flush=True)
pd.DataFrame(rows).to_csv("analysis/metrics/effects_esm2_extra.csv",index=False)
print(f"wrote analysis/metrics/effects_esm2_extra.csv ({len(rows)} rows)")
