"""Combined GPU inference (sequential, no co-run) for two figure edits:
(A) interaction CI  -> per-protein tp/fn/fp (pep+propep, corrected ±3) for 4 runs,
    pooled folds {2,5}: baseline_esm2, esm2_boundary, esmc_6b, esmc6b_boundary.
    Out: analysis/metrics/interaction_perprotein_2026.csv  (model,fold,protein,tp,fn,fp)
(B) similarity 3rd model -> per true-segment matched(±3) for esmc6b_3di_zeroctrl (proj).
    Out: analysis/similarity/seg_matched_zeroctrl.csv  (same schema as seg_matched_2026.csv)
Run: env/bin/python this.py
"""
import os, sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein, _group
dev="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")

def load(run):
    rd=f"runs/{run}"; args=load_run_args(Path(rd),SimpleNamespace(batch_size=None,device=None,seed=42))
    m=get_model(args).to(dev)
    bil=getattr(getattr(m,"feature_extractor",None),"biLSTM",None)
    if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
    m.load_state_dict(load_state_dict(os.path.join(rd,"model.pt"),dev)); m.eval()
    return args,m

# ---------- (A) interaction per-protein ----------
rowsA=[]
for run in ["2026_baseline_esm2","2026_esm2_boundary","2026_esmc_6b","2026_esmc6b_boundary"]:
    args,m=load(run)
    for fold in (2,5):
        _,_,t=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],test_partitions=[fold],device=dev)
        _,_,preds,_,_=run_dataloader(t,m,optimizer=None,do_train=False,device=dev,collect_outputs=True,desc=f"{run} f{fold}")
        data=t.dataset.data; names=t.dataset.names
        for i,pr in enumerate(preds):
            row=data.loc[names[i]]; tp=fn=fp=0
            for task,(ss,es) in [("peptides",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),
                                  ("propeptides",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
                true=[(int(a),int(b)) for a,b in (row["true_peptides"] if task=="peptides" else row["true_propeptides"])]
                pb=convert_path_to_peptide_borders(pr,start_state=ss,stop_state=es,offset=1)
                g,pm=match_protein(true,pb,3)
                tp+=sum(x["matched"] for x in g); fn+=sum(not x["matched"] for x in g); fp+=sum(not x for x in pm)
            rowsA.append(dict(model=run.replace("2026_",""),fold=fold,protein=names[i],tp=tp,fn=fn,fp=fp))
    print(f"[A done] {run}",flush=True); del m; torch.cuda.empty_cache()
dA=pd.DataFrame(rowsA); dA.to_csv("analysis/metrics/interaction_perprotein_2026.csv",index=False)
def f1arr(a): tp,fn,fp=a[["tp","fn","fp"]].sum(); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; return 2*p*r/(p+r) if p+r else 0
print("interaction abs F1:",{m:round(f1arr(dA[dA.model==m]),4) for m in dA.model.unique()})

# ---------- (B) zeroctrl per-segment matched ----------
lab2022=set(pd.read_csv("data/uniprot_2022/labeled_sequences.csv")["protein_id"])
run="esmc6b_3di_zeroctrl"; args,m=load(f"2026_{run}"); rowsB=[]
for fold in (2,5):
    _,_,test=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],test_partitions=[fold],device=dev)
    _,_,preds,_,_=run_dataloader(test,m,optimizer=None,do_train=False,device=dev,collect_outputs=True,desc=f"{run} f{fold}")
    data=test.dataset.data; names=test.dataset.names
    for i,pred in enumerate(preds):
        pid=names[i]; r=data.loc[pid]; pseq=r["sequence"]
        org=str(r.get("organism","")).split()[0] if pd.notna(r.get("organism")) else "?"
        for task,(ss,es),key in [("pep",(PEPTIDE_START_STATE,PEPTIDE_END_STATE),"true_peptides"),
                                 ("propep",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE),"true_propeptides")]:
            true=[(int(a),int(b)) for a,b in r[key]]
            if not true: continue
            pb=convert_path_to_peptide_borders(pred,start_state=ss,stop_state=es,offset=1)
            grecs,_=match_protein(true,pb,3); groups=_group(true)
            for gi,g in enumerate(groups):
                s,e=max(g,key=lambda z:z[1]-z[0]); seq=pseq[s-1:e]
                rowsB.append(dict(model=run,fold=fold,protein=pid,task=task,start=s,end=e,
                                  seq=seq,length=e-s+1,genus=org,new=pid not in lab2022,
                                  matched=int(bool(grecs[gi]["matched"]))))
dB=pd.DataFrame(rowsB); dB.to_csv("analysis/similarity/seg_matched_zeroctrl.csv",index=False)
print(f"[B done] {run}  rows={len(dB)}  recall(±3)={dB.matched.mean():.4f}  proteins={dB.protein.nunique()}")
