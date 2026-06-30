"""Re-plot data-scaling curve from CSV with HONEST framing (differential effect). No GPU.
PRES=1 -> presentation/figures only, clean title."""
import warnings; warnings.filterwarnings("ignore"); import pandas as pd, numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import PRES, title as ptitle, outdirs
df=pd.read_csv("analysis/metrics/datascale_curve.csv")
COL={"baseline":"#9aa3ad","proj":"#9a78c2","3di":"#4ca37a"}
LAB={"baseline":"ESM-2 (бейзлайн)","proj":"ESM-C 6B + boundary + gated(256), 3Di выкл","3di":"ESM-C 6B + boundary + gated(256) + 3Di"}
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12.5,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":9.5})
OUT="texs/Overleaf/figures/"
fig,ax=plt.subplots(figsize=(8.4,5.3))
dy={"baseline":0,"proj":9,"3di":-10}  # stagger delta labels so proj/3di don't overlap at the right
lastx=None
for tag in ["baseline","proj","3di"]:
    r=df[df.tag==tag].sort_values("n_train")
    ax.errorbar(r.n_train,r.f1,yerr=[r.f1-r.lo,r.hi-r.f1],fmt="o-",color=COL[tag],lw=2.0,ms=7,
                mec="white",mew=1.2,capsize=3,label=LAB[tag],zorder=3)
    xv=r.n_train.to_numpy(); fv=r.f1.to_numpy(); lastx=xv[-1]; d=fv[-1]-fv[0]
    ax.annotate(f"Δ = {d:+.02f}",(xv[-1],fv[-1]),textcoords="offset points",xytext=(8,dy[tag]),
                fontsize=9.5,color=COL[tag],weight="bold",ha="left",va="center")
ax.set_xlim(right=(lastx or 5400)+950)
ax.set_xlabel("число белков в обучающей выборке"); ax.set_ylabel("F1")
ax.set_ylim(0.44,0.72)
ax.set_title(ptitle("F1 в зависимости от числа белков в обучающей выборке\n(объединение отложенных фолдов {2} и {5}, 95 % ДИ)"),fontsize=13.5,pad=11)
ax.grid(axis="x",alpha=0.3); ax.tick_params(length=0)
ax.legend(loc="lower right",fontsize=8.5,handletextpad=0.5,labelspacing=0.3)  # bottom-right corner, below the baseline curve
fig.tight_layout()
for d in outdirs(): fig.savefig(d+"datascale_curve.png",bbox_inches="tight")
print("re-plotted datascale_curve.png (honest framing)", "(PRES)" if PRES else "")
