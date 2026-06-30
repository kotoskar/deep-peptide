"""Effect of 3Di and bond-loss as boundary-style SLOPE panels (replaces the trades
histograms for the presentation). Each panel: F1 before->after, two lines (peptides,
propeptides), marginal 95% CI on both points, paired delta label, full model list below.
ESM-C 6B. Pooled folds {2,5}, corrected +/-3. Reads clean_tol_*. No GPU.
Run from repo root: env/bin/python analysis/metrics/src/generators/effects_slopes_fig.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T=pd.read_csv("analysis/metrics/clean_tol_true.csv"); P=pd.read_csv("analysis/metrics/clean_tol_pred.csv")
rng=np.random.default_rng(42); FOLDS=[2,5]
OUTS=["analysis/metrics/figures/","texs/Overleaf/figures/","presentation/figures/"]
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":13,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":10,
 "axes.labelsize":11,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":10.5})
INK="#1a1a1a"; PEPC="#4c9be8"; PROC="#e0913f"
def ppm(m,task):
    t=T[(T.model==m)&(T.fold.isin(FOLDS))&(T.task==task)]; q=P[(P.model==m)&(P.fold.isin(FOLDS))&(P.task==task)]
    tg=t.groupby(["fold","protein"]).m3.agg(tp="sum",ntrue="size"); pg=q.groupby(["fold","protein"]).m3.agg(mp="sum",npred="size")
    d=tg.join(pg,how="outer").fillna(0.0); d["fn"]=d.ntrue-d.tp; d["fp"]=d.npred-d.mp; return d[["tp","fn","fp"]]
def f1(a):
    tp,fn,fp=a.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
def stats(before,after,task,B=3000):
    A=ppm(before,task); Bn=ppm(after,task); idx=A.index.intersection(Bn.index)
    a=A.loc[idx].to_numpy(float); b=Bn.loc[idx].to_numpy(float); n=len(idx)
    f0,f1a=f1(a),f1(b); dpt=f1a-f0
    boots=np.array([(rng.integers(0,n,n)) for _ in range(B)])
    b0=np.array([f1(a[s]) for s in boots]); b1=np.array([f1(b[s]) for s in boots])
    l0,h0=np.percentile(b0,[2.5,97.5]); l1,h1=np.percentile(b1,[2.5,97.5])
    dl,dh=np.percentile(b1-b0,[2.5,97.5])
    return (f0,l0,h0),(f1a,l1,h1),(dpt,dl,dh)
# (key, before, after, [stage labels], title, model-list footer)
PANELS=[
 ("3di","esmc6b_3di_zeroctrl","esmc6b_3di_gated_boundary",
  ["3Di занулён","+ 3Di"],"Эффект добавления 3Di",
  "ESM-C 6B + boundary + gated(256) + 3Di\nконтроль: та же модель с занулённым 3Di"),
 ("bond","esmc6b_boundary","esmc6b_boundary_bond",
  ["без bond-лосса","+ bond-лосс"],"Эффект добавления bond-лосса",
  "ESM-C 6B + boundary + bond-лосс\nконтроль: ESM-C 6B + boundary"),
]
fig,axs=plt.subplots(1,2,figsize=(12.6,5.4),sharey=True)
x=[0,1]; ymin=1.0; ymax=0.0
for ax,(key,bef,aft,stages,title,footer) in zip(axs,PANELS):
    for task,col,nm in [("pep",PEPC,"пептиды"),("propep",PROC,"пропептиды")]:
        (p0,l0,h0),(p1,l1,h1),(dt,dl,dh)=stats(bef,aft,task)
        ax.plot(x,[p0,p1],"-o",color=col,lw=2.6,ms=10,mec="white",mew=1.5,label=nm,zorder=3)
        ax.errorbar(x,[p0,p1],yerr=[[p0-l0,p1-l1],[h0-p0,h1-p1]],fmt="none",ecolor=col,
                    elinewidth=2.0,capsize=5,capthick=2.0,zorder=4)
        ax.annotate(f"Δ = {dt:+.3f}",(1,p1),textcoords="offset points",xytext=(14,0),
                    ha="left",va="center",fontsize=11,color=col,weight="bold")
        ymin=min(ymin,l0,l1); ymax=max(ymax,h0,h1)
    ax.set_xticks(x); ax.set_xticklabels(stages,fontsize=11)
    ax.set_xlim(-0.45,1.95); ax.grid(axis="x",alpha=0); ax.tick_params(length=0)
    ax.set_title(title); ax.text(0.5,-0.20,footer,transform=ax.transAxes,ha="center",va="top",
                                 fontsize=9.5,color=INK,linespacing=1.4)
axs[0].set_ylabel("F1"); axs[0].legend(loc="lower left",title="тип сегмента")
pad=0.015; axs[0].set_ylim(ymin-pad,ymax+pad)
fig.suptitle("Влияние 3Di и bond-лосса на F1 по типам сегментов",fontsize=15,weight="bold",color=INK,y=1.0)
fig.tight_layout(rect=[0,0.02,1,0.97])
for d in OUTS: fig.savefig(d+"effects_slopes.png",bbox_inches="tight")
print("wrote effects_slopes.png (3Di + bond slope panels, pep/propep)")
for key,bef,aft,stages,title,footer in PANELS:
    for task in ("pep","propep"):
        (p0,_,_),(p1,_,_),(dt,dl,dh)=stats(bef,aft,task)
        print(f"  {key:5s} {task:7s}: {p0:.3f} -> {p1:.3f}  Δ={dt:+.3f} [{dl:+.3f},{dh:+.3f}]")
