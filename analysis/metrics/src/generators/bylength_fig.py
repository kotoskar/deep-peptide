"""By-length F1/P/R for the 5 LADDER models incl. the gated-adapter (A2), corrected ±3,
pooled {2,5}, common protein set, ladder colors. Reads clean_tol_* + adapter256_seg_*. No GPU.
Writes bylength.png to analysis/metrics/figures/ + texs/Overleaf/figures/.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import MODEL_LABELS, COLORS, PRES, title as ptitle, outdirs
FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"; FOLDS=[2,5]
T=pd.read_csv(FIG.replace("figures/","")+"clean_tol_true.csv")
P=pd.read_csv(FIG.replace("figures/","")+"clean_tol_pred.csv")
AT=pd.read_csv("analysis/metrics/adapter256_seg_true.csv")
AP=pd.read_csv("analysis/metrics/adapter256_seg_pred.csv")
# unified per-segment true/pred tables (cols: model,fold,protein,task,length/plen,m3)
Tt=pd.concat([T[["model","fold","protein","task","length","m3"]],AT[["model","fold","protein","task","length","m3"]]],ignore_index=True)
Pp=pd.concat([P[["model","fold","protein","task","plen","m3"]],AP[["model","fold","protein","task","plen","m3"]]],ignore_index=True)
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12.5,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,"axes.spines.top":False,
 "axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec","xtick.color":"#555",
 "ytick.color":"#555","legend.frameon":False,"legend.fontsize":8.5}); INK="#1a1a1a"
MODELS=[(m,MODEL_LABELS[m],COLORS[m]) for m in
        ("baseline_esm2","esmc_6b","esmc6b_boundary","adapter256","esmc6b_3di_gated_boundary")]
def pset(m): return set(zip(Tt[(Tt.model==m)&(Tt.fold.isin(FOLDS))].fold,Tt[(Tt.model==m)&(Tt.fold.isin(FOLDS))].protein))
common=set.intersection(*[pset(m) for m,_,_ in MODELS])
LB=["5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-50"]
def lbin(L):
    e=[5,10,15,20,25,30,35,40,45,51]
    for i in range(len(e)-1):
        if e[i]<=L<e[i+1]: return f"{e[i]}-{e[i+1]-1}"
    return None
def f1pr(tp,fn,fp):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return (2*p*r/(p+r) if p+r else 0),p,r
def bylen(m):
    t=Tt[(Tt.model==m)&(Tt.fold.isin(FOLDS))].copy(); q=Pp[(Pp.model==m)&(Pp.fold.isin(FOLDS))].copy()
    t=t[[(f,p) in common for f,p in zip(t.fold,t.protein)]]; q=q[[(f,p) in common for f,p in zip(q.fold,q.protein)]]
    t["lb"]=t.length.map(lbin); q["lb"]=q.plen.map(lbin); res={}
    for k in LB:
        tk=t[t.lb==k]; qk=q[q.lb==k]
        if len(tk)<25: continue
        res[k]=f1pr(tk.m3.sum(),(tk.m3==0).sum(),(qk.m3==0).sum())
    return res
fig,axs=plt.subplots(1,3,figsize=(15.5,4.7),sharey=True)
keys_present=[k for k in LB if k in bylen(MODELS[0][0])]
for ax,(kind,lab,mi) in zip(axs,[("F1","F1",0),("P","Точность",1),("R","Полнота",2)]):
    for m,nm,col in MODELS:
        r=bylen(m); ks=[k for k in LB if k in r]; xs=np.arange(len(ks))
        ax.plot(xs,[r[k][mi] for k in ks],"o-",color=col,label=nm,ms=4.8,lw=1.8)
    ax.set_xticks(range(len(keys_present))); ax.set_xticklabels(keys_present,rotation=45,fontsize=8)
    ax.set_title(ptitle(lab)); ax.set_xlabel("длина сегмента (аминокислот)"); ax.set_ylim(0,1); ax.grid(axis="x",alpha=0)
axs[0].set_ylabel("значение метрики"); axs[0].legend(loc="lower center",fontsize=7.8)
fig.suptitle(ptitle("F1, точность и полнота в зависимости от длины сегмента (объединение фолдов {2} и {5})"),
             fontsize=14,weight="bold",color=INK,y=1.02)
fig.tight_layout(rect=[0,0,1,0.95])
for d in outdirs(): fig.savefig(d+"bylength.png",bbox_inches="tight")
print(f"wrote bylength.png (5 ladder models incl adapter, common n={len(common)})", "(PRES)" if PRES else "")
