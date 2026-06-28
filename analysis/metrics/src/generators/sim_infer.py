"""Per held-out true segment (folds {2,5}): seq + matched(±3), for the weak baseline
and the two TOP 6B models. Records coords/seq for max-identity-to-train join.
Out: analysis/similarity/seg_matched_2026.csv
"""
import os, sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import pandas as pd, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein, _group
device="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")
MODELS=["baseline_esm2","esmc6b_boundary","esmc6b_3di_gated_boundary"]
lab2022=set(pd.read_csv("data/uniprot_2022/labeled_sequences.csv")["protein_id"])
rows=[]
for run in MODELS:
    rd=f"runs/2026_{run}"
    args=load_run_args(Path(rd), SimpleNamespace(batch_size=None, device=None, seed=42))
    model=get_model(args).to(device)
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
            pid=names[i]; r=data.loc[pid]; pseq=r["sequence"]
            org=str(r.get("organism","")).split()[0] if pd.notna(r.get("organism")) else "?"
            for task,(ss,es),key in [("pep",(PEPTIDE_START_STATE,PEPTIDE_END_STATE),"true_peptides"),
                                     ("propep",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE),"true_propeptides")]:
                true=[(int(a),int(b)) for a,b in r[key]]
                if not true: continue
                pb=convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)
                grecs,_=match_protein(true, pb, 3); groups=_group(true)
                for gi,g in enumerate(groups):
                    s,e=max(g,key=lambda z:z[1]-z[0]); seq=pseq[s-1:e]
                    rows.append(dict(model=run,fold=fold,protein=pid,task=task,start=s,end=e,
                                     seq=seq,length=e-s+1,genus=org,new=pid not in lab2022,
                                     matched=int(bool(grecs[gi]["matched"]))))
    print(f"[done] {run}", flush=True)
df=pd.DataFrame(rows); df.to_csv("analysis/similarity/seg_matched_2026.csv",index=False)
print("wrote analysis/similarity/seg_matched_2026.csv", len(df))
print("\n=== GATE: overall recall(±3) per model (compare vs clean_tol_true) ===")
print(df.groupby("model").matched.mean().round(4).to_dict())
print("held-out proteins per model:", df.groupby("model").protein.nunique().to_dict())
