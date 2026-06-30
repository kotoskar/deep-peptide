"""Interaction figure (T1×T2×T3): slope plot plain -> +boundary-head, two embedding lines,
with PAIRED 95% delta-CI (boundary - plain) shown as the error bar on each +head point.
Reads per-protein corrected ±3 tp/fn/fp from gpu_fig_data.py (one source, common protein set).
ESM2: esm2_boundary - baseline_esm2 ;  6B: esmc6b_boundary - esmc_6b. No GPU.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import PRES, title as ptitle, outdirs
FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"; rng=np.random.default_rng(42)
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":11,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":10})
INK="#1a1a1a"
D=pd.read_csv("analysis/metrics/interaction_perprotein_2026.csv")
MS=["baseline_esm2","esm2_boundary","esmc_6b","esmc6b_boundary"]
key=lambda m:set(zip(D[D.model==m].fold,D[D.model==m].protein))
common=set.intersection(*[key(m) for m in MS])
def arr(m):
    s=D[D.model==m].copy(); s=s[[(f,p) in common for f,p in zip(s.fold,s.protein)]]
    s=s.sort_values(["fold","protein"]); return s[["tp","fn","fp"]].to_numpy(float)
def f1(a):
    tp,fn,fp=a.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
A={m:arr(m) for m in MS}
absF1={m:f1(A[m]) for m in MS}
def marg(m,B=3000):  # marginal 95% bootstrap CI on each point (both ends get a bar)
    a=A[m]; n=len(a); pt=f1(a)
    bs=np.array([f1(a[s]) for s in (rng.integers(0,n,n) for _ in range(B))])
    return pt,*np.percentile(bs,[2.5,97.5])
def paired_delta(head,plain,B=3000):  # kept for the printout (the body text cites this)
    H=A[head]; P=A[plain]; n=len(H); pt=f1(H)-f1(P)
    bs=np.array([f1(H[s])-f1(P[s]) for s in (rng.integers(0,n,n) for _ in range(B))])
    return pt,*np.percentile(bs,[2.5,97.5])
pts={
 "ESM-2 (1280)":      {"plain":"baseline_esm2","head":"esm2_boundary","col":"#9aa3ad"},
 "ESM-C 6B (2560)":   {"plain":"esmc_6b","head":"esmc6b_boundary","col":"#2f6db0"},
}
fig,ax=plt.subplots(figsize=(7.6,5.0)); x=[0,1]
for name,d in pts.items():
    p0,l0,h0=marg(d["plain"]); p1,l1,h1=marg(d["head"]); dt=p1-p0
    ax.plot(x,[p0,p1],"-o",color=d["col"],lw=2.6,ms=10,mec="white",mew=1.5,label=name,zorder=3)
    # marginal 95% CI error bar on BOTH points
    ax.errorbar(x,[p0,p1],yerr=[[p0-l0,p1-l1],[h0-p0,h1-p1]],fmt="none",ecolor=d["col"],
                elinewidth=2.0,capsize=5,capthick=2.0,zorder=4)
    ax.annotate(f"Δ = {dt:+.3f}",(1,p1),textcoords="offset points",
                xytext=(14,0),ha="left",va="center",fontsize=10.5,color=d["col"],weight="bold")
    pd_,plo,phi=paired_delta(d["head"],d["plain"])
    print(f"{name}: plain={p0:.3f} [{l0:.3f},{h0:.3f}]  head={p1:.3f} [{l1:.3f},{h1:.3f}]  Δ={dt:+.3f}  (paired CI [{plo:+.3f},{phi:+.3f}])")
ax.set_xticks(x); ax.set_xticklabels(["без boundary-головы\n(базовый CRF)","+ boundary-голова"],fontsize=10.5)
ax.set_xlim(-0.45,1.85); ax.set_ylim(0.53,0.70); ax.set_ylabel("F1")
ax.grid(axis="x",alpha=0); ax.tick_params(length=0); ax.legend(loc="upper left",title="эмбеддинг")
ax.set_title(ptitle("F1 без boundary-головы и с ней для двух эмбеддингов\n(95 % ДИ)"),fontsize=13.5,loc="center",pad=12)
fig.tight_layout()
for d in outdirs(): fig.savefig(d+"interaction.png",bbox_inches="tight")
print(f"interaction done (n_common={len(common)})", "(PRES)" if PRES else "")
