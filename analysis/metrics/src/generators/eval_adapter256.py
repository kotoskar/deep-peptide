"""Evaluate A2 (adapter-256, no struct) on folds {2,5}, corrected +/-3 matcher,
pep+propep per protein. Paired delta vs orange (esmc6b_boundary) = the ADAPTER effect.
Saves per-protein CSV for the ladder graph. GPU inference for A2 only (orange from CSV)."""
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
torch.set_float32_matmul_precision("high"); rng=np.random.default_rng(42)

def corr_counts(true,pred,tol=3):
    g,pm=match_protein(true,pred,tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)

def run_model(run_dir):
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
            rows.append({"model":"adapter256_seqonly","fold":fold,"protein":names[i],
                         "tp":acc[0],"fn":acc[1],"fp":acc[2]})
        del model; torch.cuda.empty_cache()
    return pd.DataFrame(rows)

def f1(a):
    tp,fn,fp=a.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0

A2=run_model("runs/2026_esmc6b_adapter256_seqonly")
A2.to_csv("analysis/metrics/adapter256_perprotein_2026.csv",index=False)
I=pd.read_csv("analysis/metrics/interaction_perprotein_2026.csv")
OR=I[I.model=="esmc6b_boundary"]      # orange = boundary, no adapter
SixB=I[I.model=="esmc_6b"]            # plain 6B (ladder rung below orange)

def keyed(df): return {(f,p):(tp,fn,fp) for f,p,tp,fn,fp in zip(df.fold,df.protein,df.tp,df.fn,df.fp)}
a2k=keyed(A2); ork=keyed(OR); sbk=keyed(SixB)

def paired(newk, refk, B=3000):
    common=sorted(set(newk)&set(refk))
    an=np.array([newk[k] for k in common],float); ar=np.array([refk[k] for k in common],float)
    fn_, fr_=f1(an), f1(ar); n=len(common)
    bs=np.array([f1(an[s])-f1(ar[s]) for s in (rng.integers(0,n,n) for _ in range(B))])
    lo,hi=np.percentile(bs,[2.5,97.5])
    return fn_, fr_, fn_-fr_, lo, hi, n

print("\n===== ADAPTER ISOLATION (corrected +/-3, pooled {2,5}) =====")
fa,fo,d,lo,hi,n=paired(a2k,ork)
print(f"orange  esmc6b_boundary (no adapter) F1 = {fo:.4f}")
print(f"A2      adapter-256, no struct       F1 = {fa:.4f}   (n_common={n})")
print(f"ADAPTER effect  A2 - orange = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
fa2,fs,d2,lo2,hi2,n2=paired(a2k,sbk)
print(f"\n(context) plain 6B F1 = {fs:.4f} ; A2 - 6B = {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")
with open("analysis/metrics/adapter256_result.txt","w") as f:
    f.write(f"orange={fo:.4f}\nA2_adapter256={fa:.4f}\nadapter_delta={d:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}\n")
print("\nwrote analysis/metrics/adapter256_perprotein_2026.csv + result.txt")
