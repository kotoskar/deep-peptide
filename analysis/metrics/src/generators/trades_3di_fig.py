"""3Di trade-off figure (single panel, no bond): ΔP/ΔR by segment type,
3Di-on (esmc6b_3di_gated_boundary) vs 3Di-zeroed control (esmc6b_3di_zeroctrl),
pooled folds {2,5}, paired 95% CI. Writes trades_3di.png to both figure dirs. No GPU.
Run from repo root: env/bin/python analysis/metrics/src/generators/trades_3di_fig.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T=pd.read_csv("analysis/metrics/clean_tol_true.csv"); P=pd.read_csv("analysis/metrics/clean_tol_pred.csv")
rng=np.random.default_rng(42); FOLDS=[2,5]; FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12.5,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":9})
INK="#1a1a1a"; RED="#d1495b"; BLU="#3c7bb0"
WIN,ZERO="esmc6b_3di_gated_boundary","esmc6b_3di_zeroctrl"
def metric(arr,kind):
    tp,fn,fp=arr.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return {"p":p,"r":r}[kind]
def ppm(m,task):
    t=T[(T.model==m)&(T.fold.isin(FOLDS))&(T.task==task)]; q=P[(P.model==m)&(P.fold.isin(FOLDS))&(P.task==task)]
    tg=t.groupby(["fold","protein"]).m3.agg(tp="sum",ntrue="size"); pg=q.groupby(["fold","protein"]).m3.agg(mp="sum",npred="size")
    d=tg.join(pg,how="outer").fillna(0.0); d["fn"]=d.ntrue-d.tp; d["fp"]=d.npred-d.mp; return d[["tp","fn","fp"]]
def paired(kind,task,B=3000):
    A=ppm(WIN,task); Bn=ppm(ZERO,task); idx=A.index.intersection(Bn.index)
    A=A.loc[idx].to_numpy(); Bm=Bn.loc[idx].to_numpy(); n=len(idx)
    pt=metric(A,kind)-metric(Bm,kind)
    bs=np.array([metric(A[s],kind)-metric(Bm[s],kind) for s in (rng.integers(0,n,n) for _ in range(B))])
    return pt,*np.percentile(bs,[2.5,97.5])
fig,ax=plt.subplots(figsize=(7.4,4.8)); x=np.arange(2); w=0.32
for kind,col,nm,dx in [("p",RED,"Δ точность",-w/2),("r",BLU,"Δ полнота",w/2)]:
    pts=[];los=[];his=[]
    for task in ("pep","propep"):
        pt,lo,hi=paired(kind,task); pts.append(pt);los.append(pt-lo);his.append(hi-pt)
        ax.annotate(f"{pt:+.03f}",(x[("pep","propep").index(task)]+dx,pt),textcoords="offset points",
                    xytext=(0, 6 if pt>=0 else -12),ha="center",fontsize=8.2,color=col,weight="bold")
    ax.bar(x+dx,pts,w,color=col,label=nm,zorder=3,edgecolor="white",linewidth=1)
    ax.errorbar(x+dx,pts,yerr=[los,his],fmt="none",ecolor=INK,elinewidth=1.3,capsize=0,zorder=4)
ax.axhline(0,color="#999",lw=1); ax.set_xticks(x); ax.set_xticklabels(["пептиды","пропептиды"])
ax.set_ylabel("разница метрики   (3Di вкл − 3Di выкл)"); ax.grid(axis="x",alpha=0); ax.tick_params(length=0)
ax.legend(loc="lower left")
ax.set_title("Эффект добавления 3Di по типам сегментов\n(объединение фолдов {2} и {5}, 95 % ДИ)",fontsize=12.5,pad=11)
fig.tight_layout()
fig.savefig(FIG+"trades_3di.png",bbox_inches="tight"); fig.savefig(OUT+"trades_3di.png",bbox_inches="tight")
print("wrote trades_3di.png (3Di only, no bond)")
for task in ("pep","propep"):
    dp=paired("p",task); dr=paired("r",task)
    print(f"  {task:6s}  ΔP={dp[0]:+.3f} [{dp[1]:+.3f},{dp[2]:+.3f}]  ΔR={dr[0]:+.3f} [{dr[1]:+.3f},{dr[2]:+.3f}]")
