"""A2 (adapter-256) tolerance curve: corrected matcher at tol 0,1,2,3, pooled {2,5},
pep+propep per protein. One inference pass, 4 tolerance evals. Saves per-protein CSV
(tp/fn/fp per tol) for the iterative tolerance graph; prints F1 curve + retention."""
import os, sys; sys.path.insert(0, os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")
TOLS=[0,1,2,3]
def counts(true,pred,tol):
    g,pm=match_protein(true,pred,tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)

run_dir="runs/2026_esmc6b_adapter256_seqonly"
args=load_run_args(Path(run_dir),SimpleNamespace(batch_size=None,device=None,seed=42))
rows=[]
for fold in (2,5):
    _,_,test=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],
                             test_partitions=[fold],device=device)
    model=get_model(args).to(device)
    bil=getattr(getattr(model,"feature_extractor",None),"biLSTM",None)
    if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
    model.load_state_dict(load_state_dict(os.path.join(run_dir,"model.pt"),device)); model.eval()
    _,_,preds,_,_=run_dataloader(test,model,optimizer=None,do_train=False,device=device,
                                 collect_outputs=True,desc=f"A2:f{fold}")
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
        rows.append({"model":"adapter256_seqonly","fold":fold,"protein":names[i],
                     **{f"tp{t}":acc[t][0] for t in TOLS},**{f"fn{t}":acc[t][1] for t in TOLS},
                     **{f"fp{t}":acc[t][2] for t in TOLS}})
    del model; torch.cuda.empty_cache()
df=pd.DataFrame(rows); df.to_csv("analysis/metrics/adapter256_tol_perprotein.csv",index=False)
def f1(a):
    tp,fn,fp=a.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
print("\n=== A2 adapter-256 tolerance curve (pooled {2,5}, corrected) ===")
fvals={}
for t in TOLS:
    a=df[[f"tp{t}",f"fn{t}",f"fp{t}"]].to_numpy(float); fvals[t]=f1(a)
    print(f"  F1@±{t} = {fvals[t]:.4f}")
print(f"  retention (±0/±3) = {fvals[0]/fvals[3]:.3f}")
print("wrote analysis/metrics/adapter256_tol_perprotein.csv")
