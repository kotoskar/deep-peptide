"""Two-panel length figure for the presentation:
 (a) per-fold segment-length profile (curves) + whole-dataset line;
 (b) F1 vs segment length for the 5 ladder models (no P/R), unified full-mod legend.
Pooled folds {2,5}, corrected +/-3, common protein set. Reads clean_tol_* +
adapter256_seg_* + data/uniprot_2026. No GPU.
Run from repo root: env/bin/python analysis/metrics/src/generators/length_profile_f1_fig.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, re; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import MODEL_LABELS, COLORS, RC, INK
plt.rcParams.update(RC)
OUTS=["analysis/metrics/figures/","texs/Overleaf/figures/","presentation/figures/"]
FOLDS=[2,5]
LBINS=[5,10,15,20,25,30,35,40,45,51]; KEYS=[f"{LBINS[i]}-{LBINS[i+1]-1}" for i in range(len(LBINS)-1)]
def lbin(L):
    for i in range(len(LBINS)-1):
        if LBINS[i]<=L<LBINS[i+1]: return KEYS[i]
    return None
def segs(s): return [(int(a),int(b)) for a,b in re.findall(r"\((\d+)-(\d+)\)",s)] if isinstance(s,str) else []

# ---- (a) per-fold length profile ----
d26=pd.read_csv("data/uniprot_2026/labeled_sequences.csv")
asg=pd.read_csv("data/uniprot_2026/graphpart_assignments.csv").rename(columns={"AC":"protein_id","cluster":"fold"})
df=d26.merge(asg,on="protein_id")
rows=[]
for _,r in df.iterrows():
    for col in ("coordinates","propeptide_coordinates"):
        for a,b in segs(r.get(col)):
            lb=lbin(b-a+1)
            if lb: rows.append((int(r["fold"]),lb))
S=pd.DataFrame(rows,columns=["fold","lb"])
piv=S.groupby(["fold","lb"]).size().unstack(fill_value=0); piv=piv.div(piv.sum(1),axis=0)[KEYS]
glob=(S.groupby("lb").size()/len(S))[KEYS]

# ---- (b) F1 by length, 5 ladder models ----
T=pd.read_csv("analysis/metrics/clean_tol_true.csv"); P=pd.read_csv("analysis/metrics/clean_tol_pred.csv")
AT=pd.read_csv("analysis/metrics/adapter256_seg_true.csv"); AP=pd.read_csv("analysis/metrics/adapter256_seg_pred.csv")
Tt=pd.concat([T[["model","fold","protein","task","length","m3"]],AT[["model","fold","protein","task","length","m3"]]],ignore_index=True)
Pp=pd.concat([P[["model","fold","protein","task","plen","m3"]],AP[["model","fold","protein","task","plen","m3"]]],ignore_index=True)
MODELS=["baseline_esm2","esmc_6b","esmc6b_boundary","adapter256","esmc6b_3di_gated_boundary"]
def pset(m): s=Tt[(Tt.model==m)&(Tt.fold.isin(FOLDS))]; return set(zip(s.fold,s.protein))
common=set.intersection(*[pset(m) for m in MODELS])
def f1_(tp,fn,fp):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
def bylen(m):
    t=Tt[(Tt.model==m)&(Tt.fold.isin(FOLDS))].copy(); q=Pp[(Pp.model==m)&(Pp.fold.isin(FOLDS))].copy()
    t=t[[(f,p) in common for f,p in zip(t.fold,t.protein)]]; q=q[[(f,p) in common for f,p in zip(q.fold,q.protein)]]
    t["lb"]=t.length.map(lbin); q["lb"]=q.plen.map(lbin); res={}
    for k in KEYS:
        tk=t[t.lb==k]; qk=q[q.lb==k]
        if len(tk)<25: continue
        res[k]=f1_(tk.m3.sum(),(tk.m3==0).sum(),(qk.m3==0).sum())
    return res

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14.2,5.2),gridspec_kw={"width_ratios":[1,1.05]})
x=np.arange(len(KEYS))
cmap=plt.cm.viridis(np.linspace(0.15,0.85,len(piv.index)))
ax1.plot(x,glob.values*100,"o-",color=INK,lw=2.6,ms=6,label="весь датасет",zorder=5)
for i,f in enumerate(piv.index):
    ax1.plot(x,piv.loc[f].values*100,"-",color=cmap[i],lw=1.4,alpha=0.85,label=f"фолд {f}")
ax1.set_xticks(x); ax1.set_xticklabels(KEYS,rotation=45,fontsize=8.5); ax1.set_xlabel("длина сегмента (аминокислот)")
ax1.set_ylabel("доля сегментов, %"); ax1.set_title("Профиль длин по фолдам"); ax1.legend(ncol=2,fontsize=8.5); ax1.grid(axis="x",alpha=0)
ref=bylen(MODELS[0]); keys_present=[k for k in KEYS if k in ref]
for m in MODELS:
    r=bylen(m); ks=[k for k in KEYS if k in r]; xs=np.arange(len(ks))
    ax2.plot(xs,[r[k] for k in ks],"o-",color=COLORS[m],label=MODEL_LABELS[m],ms=5,lw=1.9)
ax2.set_xticks(range(len(keys_present))); ax2.set_xticklabels(keys_present,rotation=45,fontsize=8.5)
ax2.set_xlabel("длина сегмента (аминокислот)"); ax2.set_ylabel("F1"); ax2.set_ylim(0,1)
ax2.set_title("F1 в зависимости от длины сегмента"); ax2.grid(axis="x",alpha=0); ax2.legend(loc="lower center",fontsize=8.2)
fig.suptitle("Распределение длин сегментов по фолдам и качество модели по длине",fontsize=15,weight="bold",color=INK,y=1.01)
fig.tight_layout(rect=[0,0,1,0.96])
for d in OUTS: fig.savefig(d+"length_profile_f1.png",bbox_inches="tight")
print(f"wrote length_profile_f1.png (common n={len(common)})")
