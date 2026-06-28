"""Free win: close the ESM2 boundary-head link. Inference only (no training).
Run baseline_esm2 and esm2_boundary on pooled folds {2,5}, corrected +/-3 matcher,
pep+propep, common protein set -> pooled F1 each + paired bootstrap delta.
"""
import os, sys, csv; sys.path.insert(0, os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import numpy as np, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")
rng=np.random.default_rng(42)

def corr_counts(true,pred,tol=3):
    g,pm=match_protein(true,pred,tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)

def run_model(run_dir):
    args=load_run_args(Path(run_dir),SimpleNamespace(batch_size=None,device=None,seed=42))
    per={}  # protein -> [tp,fn,fp]
    for fold in (2,5):
        _,_,test=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],
                                 test_partitions=[fold],device=device)
        model=get_model(args).to(device)
        bil=getattr(getattr(model,"feature_extractor",None),"biLSTM",None)
        if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
        model.load_state_dict(load_state_dict(os.path.join(run_dir,"model.pt"),device)); model.eval()
        _,_,preds,_,_=run_dataloader(test,model,optimizer=None,do_train=False,device=device,
                                     collect_outputs=True,desc=os.path.basename(run_dir)+f":f{fold}")
        data=test.dataset.data; names=test.dataset.names
        for i,pred in enumerate(preds):
            row=data.loc[names[i]]; acc=[0,0,0]
            for task,(ss,es) in [("peptides",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),
                                 ("propeptides",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
                true=[(int(a),int(b)) for a,b in (row["true_peptides"] if task=="peptides" else row["true_propeptides"])]
                pb=convert_path_to_peptide_borders(pred,start_state=ss,stop_state=es,offset=1)
                c=corr_counts(true,pb,3)
                for k in range(3): acc[k]+=c[k]
            per[(fold,names[i])]=acc
        del model; torch.cuda.empty_cache()
    return per

def f1(arr):
    tp,fn,fp=arr.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0

B=run_model("runs/2026_baseline_esm2"); H=run_model("runs/2026_esm2_boundary")
common=sorted(set(B)&set(H))
ab=np.array([B[k] for k in common],float); ah=np.array([H[k] for k in common],float)
f0=f1(ab); f1h=f1(ah); n=len(common)
bs=np.array([f1(ah[s])-f1(ab[s]) for s in (rng.integers(0,n,n) for _ in range(3000))])
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"\n=== ESM2 boundary-head link (pooled {{2,5}}, corrected +/-3, n={n}) ===")
print(f"baseline_esm2  F1 = {f0:.4f}")
print(f"esm2_boundary  F1 = {f1h:.4f}")
print(f"paired Delta(+boundary) = {f1h-f0:+.4f}  [{lo:+.4f},{hi:+.4f}]")
with open("analysis/metrics/esm2_boundary_result.txt","w") as f:
    f.write(f"baseline_esm2={f0:.4f}\nesm2_boundary={f1h:.4f}\ndelta={f1h-f0:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}\n")
print("DONE")
