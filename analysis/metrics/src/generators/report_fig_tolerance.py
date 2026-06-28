"""Tolerance figure (T8), clean style: F1 vs match tolerance ±3→±0, pooled {2,5}, 4 key models.
Shows the boundary head keeps borders sharper (slower F1 decay). CPU only (reads clean_tol_*).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T=pd.read_csv("analysis/metrics/clean_tol_true.csv"); P=pd.read_csv("analysis/metrics/clean_tol_pred.csv")
FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"; FOLDS=[2,5]
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":9.5})
INK="#1a1a1a"; MUTE="#8a9099"
SEL=[("baseline_esm2","ESM-2 (база)","#9aa3ad","o"),
     ("esmc_6b","ESM-C 6B (без головы)","#5b8bc0","s"),
     ("esmc6b_boundary","6B + boundary (без gated)","#e0913f","^"),
     ("esmc6b_3di_zeroctrl","6B + boundary + gated, 3Di выкл","#9a78c2","v"),
     ("esmc6b_3di_gated_boundary","6B + boundary + gated + 3Di","#4ca37a","D")]
def common(models,fold):
    s=None
    for m in models:
        ts=set((f,p) for f,p in zip(T[T.model==m].fold,T[T.model==m].protein) if f==fold)
        ps=set((f,p) for f,p in zip(P[P.model==m].fold,P[P.model==m].protein) if f==fold)
        u=ts|ps; s=u if s is None else s&u
    return s
cset=set()
for f in FOLDS: cset|=common([m for m,_,_,_ in SEL],f)
def f1(model,tol):
    t=T[(T.model==model)]; q=P[(P.model==model)]
    t=t[[ (f,p) in cset for f,p in zip(t.fold,t.protein)]]; q=q[[ (f,p) in cset for f,p in zip(q.fold,q.protein)]]
    tp=t[f"m{tol}"].sum(); fn=(t[f"m{tol}"]==0).sum(); fp=(q[f"m{tol}"]==0).sum()
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; return 2*p*r/(p+r) if p+r else 0
tols=[3,2,1,0]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.6))
for m,nm,col,mk in SEL:
    ys=[f1(m,t) for t in tols]; ax1.plot(range(4),ys,mk+"-",color=col,label=nm,ms=6,lw=1.9,mec="white",mew=1)
    ret=[y/ys[0] for y in ys]; ax2.plot(range(4),ret,mk+"-",color=col,label=nm,ms=6,lw=1.9,mec="white",mew=1)
    ax2.annotate(f"{ret[-1]:.2f}",(3,ret[-1]),textcoords="offset points",xytext=(8,0),fontsize=8.5,color=col,weight="bold")
for ax in (ax1,ax2):
    ax.set_xticks(range(4)); ax.set_xticklabels([f"±{t}" for t in tols]); ax.grid(axis="x",alpha=0)
    ax.set_xlabel("допуск совпадения границ"); ax.tick_params(length=0)
ax1.set_ylabel("F1"); ax1.set_title("(а) F1 при ужесточении допуска",loc="center"); ax1.legend(loc="lower left")
ax2.set_ylabel("доля сохранённого F1  (F1 / F1 при ±3)"); ax2.set_title("(б) Доля сохранённого F1 (F1@±0 / F1@±3)",loc="center")
fig.suptitle("F1 при ужесточении допуска совпадения границ (±3 → ±0), объединение фолдов {2} и {5}",
             fontsize=14,weight="bold",color=INK,y=1.02)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG+"tolerance.png",bbox_inches="tight"); fig.savefig(OUT+"tolerance.png",bbox_inches="tight"); print("tolerance done")
for m,nm,_,_ in SEL: print(f"  {nm:34s} F1@±3={f1(m,3):.3f}  ±0={f1(m,0):.3f}  retention={f1(m,0)/f1(m,3):.2f}")
