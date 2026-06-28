"""T0 data-scaling curve: F1 vs train size on pooled {2,5}, corrected ±3, 3 models.
Lower points = scale_*_{40,55,70,85}; 100% point = existing 2026_* (REUSES). GPU inference.
Writes analysis/metrics/figures/datascale_curve.png + analysis/metrics/datascale_curve.csv.
Run: env/bin/python this.py
"""
import os,sys,csv,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import numpy as np, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders, PEPTIDE_START_STATE,
    PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein
dev="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision("high")
rng=np.random.default_rng(42)

# (tag, label, color) and per-frac run dir + n_train (from manifest)
MODELS={
 "baseline":("ESM-2 (базовая)","#9aa3ad",
   {40:("runs/scale_baseline_40",2102),55:("runs/scale_baseline_55",2914),70:("runs/scale_baseline_70",3679),
    85:("runs/scale_baseline_85",4527),100:("runs/2026_baseline_esm2",5314)}),
 "proj":("ESM-C 6B + boundary (без 3Di)","#9a78c2",
   {40:("runs/scale_proj_40",2131),55:("runs/scale_proj_55",2952),70:("runs/scale_proj_70",3726),
    85:("runs/scale_proj_85",4586),100:("runs/2026_esmc6b_3di_zeroctrl",5384)}),
 "3di":("ESM-C 6B + boundary + 3Di","#4ca37a",
   {40:("runs/scale_3di_40",2088),55:("runs/scale_3di_55",2897),70:("runs/scale_3di_70",3660),
    85:("runs/scale_3di_85",4504),100:("runs/2026_esmc6b_3di_gated_boundary",5287)}),
}
def corr(true,pred):
    g,pm=match_protein(true,pred,3)
    return sum(x["matched"] for x in g),sum(not x["matched"] for x in g),sum(not x for x in pm)
def evalF1(rundir):
    args=load_run_args(Path(rundir),SimpleNamespace(batch_size=None,device=None,seed=42))
    m=get_model(args).to(dev); bil=getattr(getattr(m,"feature_extractor",None),"biLSTM",None)
    if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
    m.load_state_dict(load_state_dict(os.path.join(rundir,"model.pt"),dev)); m.eval()
    per=[]
    for fold in (2,5):
        _,_,t=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],test_partitions=[fold],device=dev)
        _,_,preds,_,_=run_dataloader(t,m,optimizer=None,do_train=False,device=dev,collect_outputs=True,desc=os.path.basename(rundir)+f" f{fold}")
        data=t.dataset.data; names=t.dataset.names
        for i,pr in enumerate(preds):
            row=data.loc[names[i]]; tp=fn=fp=0
            for task,(ss,es) in [("peptides",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),("propeptides",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
                true=[(int(a),int(b)) for a,b in (row["true_peptides"] if task=="peptides" else row["true_propeptides"])]
                pb=convert_path_to_peptide_borders(pr,start_state=ss,stop_state=es,offset=1)
                c=corr(true,pb); tp+=c[0]; fn+=c[1]; fp+=c[2]
            per.append((tp,fn,fp))
    arr=np.array(per,float); TP,FN,FP=arr.sum(0)
    p=TP/(TP+FP) if TP+FP else 0; r=TP/(TP+FN) if TP+FN else 0; f1=2*p*r/(p+r) if p+r else 0
    # bootstrap CI over proteins
    n=len(arr); bs=[]
    for _ in range(1500):
        s=arr[rng.integers(0,n,n)].sum(0); tp,fn,fp=s
        pp=tp/(tp+fp) if tp+fp else 0; rr=tp/(tp+fn) if tp+fn else 0; bs.append(2*pp*rr/(pp+rr) if pp+rr else 0)
    lo,hi=np.percentile(bs,[2.5,97.5])
    return f1,p,r,lo,hi,n

rows=[]
for tag,(label,col,fracs) in MODELS.items():
    for frac in sorted(fracs):
        rd,ntr=fracs[frac]; f1,p,r,lo,hi,n=evalF1(rd)
        rows.append(dict(tag=tag,label=label,frac=frac,n_train=ntr,f1=round(f1,4),p=round(p,4),r=round(r,4),lo=round(lo,4),hi=round(hi,4),n_eval=n))
        print(f"  {tag:9s} {frac:3d}% n_train={ntr:5d}  F1={f1:.3f} [{lo:.3f},{hi:.3f}]  P={p:.3f} R={r:.3f}",flush=True)
with open("analysis/metrics/datascale_curve.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["tag","label","frac","n_train","f1","p","r","lo","hi","n_eval"]); w.writeheader(); w.writerows(rows)
print("wrote analysis/metrics/datascale_curve.csv")

# ---- plot (clean style) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12.5,"axes.titleweight":"bold","axes.titlelocation":"left","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":9.5})
fig,ax=plt.subplots(figsize=(8.2,5.2))
for tag,(label,col,fracs) in MODELS.items():
    rr=[x for x in rows if x["tag"]==tag]; rr.sort(key=lambda z:z["n_train"])
    xs=[z["n_train"] for z in rr]; ys=[z["f1"] for z in rr]
    lo=[z["f1"]-z["lo"] for z in rr]; hi=[z["hi"]-z["f1"] for z in rr]
    ax.errorbar(xs,ys,yerr=[lo,hi],fmt="o-",color=col,lw=2.0,ms=7,mec="white",mew=1.2,capsize=3,label=label,zorder=3)
ax.set_xlabel("число обучающих белков"); ax.set_ylabel("F1 на объединённых фолдах {2}+{5} (±3)")
ax.set_title("Качество растёт с объёмом данных и не выходит на плато",pad=11)
ax.grid(axis="x",alpha=0.3); ax.tick_params(length=0); ax.legend(loc="lower right",title="модель")
fig.text(0.012,-0.02,"кривые ещё идут вверх к 100 % — данные не исчерпаны; прирост от данных сопоставим с архитектурным",
         ha="left",fontsize=8.6,color="#8a9099",transform=ax.transAxes)
fig.tight_layout(); fig.savefig("analysis/metrics/figures/datascale_curve.png",bbox_inches="tight")
print("wrote datascale_curve.png")
