"""Extended tolerance data for the ladder_tol figure: per-protein tp/fn/fp at
tolerances 0..5 (pep+propep, corrected match_protein), pooled folds {2,5}, for the
6 ladder-tol models. Writes analysis/metrics/ladder_tol_ext.csv. GPU inference.
Run from repo root: env/bin/python analysis/metrics/src/generators/ladder_tol_ext_eval.py
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
TOLS=[0,1,2,3,4,5]
MODELS=[("baseline_esm2","runs/2026_baseline_esm2"),
        ("esmc_6b","runs/2026_esmc_6b"),
        ("esmc6b_boundary","runs/2026_esmc6b_boundary"),
        ("adapter256","runs/2026_esmc6b_adapter256_seqonly"),
        ("esmc6b_3di_gated_boundary","runs/2026_esmc6b_3di_gated_boundary"),
        ("esmc6b_3di_nocompress","runs/2026_esmc6b_3di_nocompress")]
def counts(true,pred,tol):
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
            row=data.loc[names[i]]; acc={t:[0,0,0] for t in TOLS}
            for task,(ss,es) in [("peptides",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),
                                 ("propeptides",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
                true=[(int(a),int(b)) for a,b in (row["true_peptides"] if task=="peptides" else row["true_propeptides"])]
                pb=convert_path_to_peptide_borders(pred,start_state=ss,stop_state=es,offset=1)
                for t in TOLS:
                    c=counts(true,pb,t)
                    for k in range(3): acc[t][k]+=c[k]
            rows.append({"model":name,"fold":fold,"protein":names[i],
                         **{f"tp{t}":acc[t][0] for t in TOLS},**{f"fn{t}":acc[t][1] for t in TOLS},
                         **{f"fp{t}":acc[t][2] for t in TOLS}})
        del model; torch.cuda.empty_cache()
    print(f"[done] {name}", flush=True)
pd.DataFrame(rows).to_csv("analysis/metrics/ladder_tol_ext.csv",index=False)
print(f"wrote analysis/metrics/ladder_tol_ext.csv ({len(rows)} rows, {len(MODELS)} models, tols {TOLS})")
