"""Border-sharpness vs data, WITH bootstrap CIs (reads per-protein CSV). No GPU.
(a) ESM-C 6B (proj): F1 at ±3 and ±0 vs n_train, with 95% CI.
(b) retention F1@±0/F1@±3 vs n_train, 3 models, with 95% CI.
Descriptive titles only. Run: env/bin/python this.py
"""
import warnings; warnings.filterwarnings("ignore"); import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=pd.read_csv("analysis/metrics/datascale_tol_perprotein.csv")
rng=np.random.default_rng(42)
COL={"baseline":"#9aa3ad","proj":"#9a78c2","3di":"#4ca37a"}
LAB={"baseline":"ESM-2 (базовая)","proj":"ESM-C 6B + boundary + gated (3Di выкл)","3di":"ESM-C 6B + boundary + gated + 3Di"}
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":9})
def f1(tp,fn,fp):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; return 2*p*r/(p+r) if p+r else 0
def boot(sub,B=2000):
    a3=sub[["tp3","fn3","fp3"]].to_numpy(float); a0=sub[["tp0","fn0","fp0"]].to_numpy(float); n=len(sub)
    pt3=f1(*a3.sum(0)); pt0=f1(*a0.sum(0)); ptr=pt0/pt3 if pt3 else 0
    b3=np.empty(B); b0=np.empty(B); br=np.empty(B)
    for i in range(B):
        s=rng.integers(0,n,n); s3=a3[s].sum(0); s0=a0[s].sum(0)
        x3=f1(*s3); x0=f1(*s0); b3[i]=x3; b0[i]=x0; br[i]=x0/x3 if x3 else 0
    q=lambda v:np.percentile(v,[2.5,97.5])
    return (pt3,*q(b3)),(pt0,*q(b0)),(ptr,*q(br))
rec={}
for (tag,frac),sub in D.groupby(["tag","frac"]):
    ntr=int(sub.n_train.iloc[0]); rec[(tag,frac)]=(ntr,*boot(sub))
def series(tag):
    fr=sorted(f for (t,f) in rec if t==tag);
    return [rec[(tag,f)] for f in fr]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.2,4.9))
# (a) proj F1@3 vs F1@0 with CI
s=series("proj"); xs=[r[0] for r in s]
for idx,(lab,col,ls,mk) in [(1,("допуск ±3","#9a78c2","-","o")),(2,("допуск ±0","#c0392b","--","s"))]:
    pt=[r[idx][0] for r in s]; lo=[r[idx][1] for r in s]; hi=[r[idx][2] for r in s]
    ax1.plot(xs,pt,mk,linestyle=ls,color=col,lw=2,ms=7,mec="white",mew=1.2,label=lab)
    ax1.fill_between(xs,lo,hi,color=col,alpha=0.15,linewidth=0)
    d=pt[-1]-pt[0]  # delta 40%->100%
    ax1.annotate(f"Δ = {d:+.02f}",(xs[-1],pt[-1]),textcoords="offset points",xytext=(6,-2),
                 fontsize=9.5,color=col,weight="bold",ha="left",va="center")
ax1.set_xlabel("число белков в обучающей выборке"); ax1.set_ylabel("F1"); ax1.legend(loc="center right")
ax1.set_xlim(right=xs[-1]+700)
ax1.set_title("(а) ESM-C 6B: F1 при допусках ±3 и ±0"); ax1.grid(axis="x",alpha=0.3); ax1.tick_params(length=0)
# (b) retention 3 models with CI
lastx=None; dy={"baseline":0,"proj":9,"3di":-9}  # stagger labels so proj/3di don't overlap
for tag in ["baseline","proj","3di"]:
    s=series(tag); xs=[r[0] for r in s]; lastx=xs[-1]; pt=[r[3][0] for r in s]; lo=[r[3][1] for r in s]; hi=[r[3][2] for r in s]
    ax2.plot(xs,pt,"o-",color=COL[tag],lw=2,ms=7,mec="white",mew=1.2,label=LAB[tag])
    ax2.fill_between(xs,lo,hi,color=COL[tag],alpha=0.13,linewidth=0)
    d=pt[-1]-pt[0]
    ax2.annotate(f"Δ = {d:+.02f}",(xs[-1],pt[-1]),textcoords="offset points",xytext=(7,dy[tag]),
                 fontsize=9,color=COL[tag],weight="bold",ha="left",va="center")
ax2.set_xlim(right=(lastx or 5400)+900)
ax2.set_xlabel("число белков в обучающей выборке"); ax2.set_ylabel("доля сохранённого F1  (F1@±0 / F1@±3)")
ax2.set_title("(б) Доля сохранённого F1 при точном совпадении"); ax2.legend(loc="lower right")
ax2.grid(axis="x",alpha=0.3); ax2.tick_params(length=0)
fig.suptitle("F1 при допусках ±3 и ±0 в зависимости от числа белков в обучающей выборке (объединение фолдов {2} и {5}, 95 % ДИ)",
             fontsize=13.5,weight="bold",color="#1a1a1a",y=1.02)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("analysis/metrics/figures/datascale_tolerance.png",bbox_inches="tight"); fig.savefig("texs/Overleaf/figures/datascale_tolerance.png",bbox_inches="tight")
print("wrote datascale_tolerance.png (with CIs)")
