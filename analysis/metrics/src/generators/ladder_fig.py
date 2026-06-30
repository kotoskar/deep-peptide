"""Iterative ladder for the presentation: база→6B→boundary→gated-адаптер(256)→3Di,
corrected ±3, pooled {2,5}, on ONE common protein set. 'сжатие' = control (nocompress≈green).
Renders: climbing F1 ladder + matching tolerance pair, plus incremental build-up frames.
Reads clean_tol_* (base/6B/orange/green/nocompress) + adapter256_tol_perprotein (A2). No GPU.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import PRES, title as ptitle
rng=np.random.default_rng(42); FOLDS=[2,5]
ALLTOLS=[0,1,2,3,4,5]; TOLPLOT=[5,4,3,2,1,0]; REFTOL=3  # tol panel shows ±5..±0; retention normalized to ±3
PRES_DIR="presentation/figures/"
OUT=("presentation/figures/ladder/" if PRES else "analysis/metrics/figures/ladder/"); os.makedirs(OUT,exist_ok=True)
REPORT="texs/Overleaf/figures/"
def final_dirs(name):  # where the final (upto==6) figure lands
    return [PRES_DIR+name, OUT+name] if PRES else [OUT+name, REPORT+name]
E=pd.read_csv("analysis/metrics/ladder_tol_ext.csv")  # per-protein tp/fn/fp at tols 0..5, 6 models
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12.5,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":11,"axes.edgecolor":"#bbb","axes.linewidth":1.0,"axes.spines.top":False,
 "axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec","xtick.color":"#555",
 "ytick.color":"#555","legend.frameon":False,"legend.fontsize":9.5}); INK="#1a1a1a"; MUTE="#8a9099"

def ext_counts(model):  # model -> {(fold,protein): {tol:[tp,fn,fp]}} from ladder_tol_ext.csv (tols 0..5)
    s=E[(E.model==model)&(E.fold.isin(FOLDS))]
    return {(r.fold,r.protein):{t:[r[f"tp{t}"],r[f"fn{t}"],r[f"fp{t}"]] for t in ALLTOLS} for _,r in s.iterrows()}

# rungs in climbing order + the control
RUNGS=[("baseline_esm2","Бейзлайн: ESM-2","#9aa3ad"),
       ("esmc_6b","ESM-2 → ESM-C 6B","#5b8bc0"),
       ("esmc6b_boundary","+ boundary-голова","#e0913f"),
       ("adapter256","+ gated-адаптер (256)","#9a78c2"),
       ("esmc6b_3di_gated_boundary","+ 3Di","#4ca37a")]
DISP=["Бейзлайн\nESM-2","ESM-2 →\nESM-C 6B","+ boundary-\nголова","+ gated-адаптер\n(256)","+ 3Di"]
CTRL=("esmc6b_3di_nocompress","без сжатия (2560) — контроль","#5fae93")
C={m:ext_counts(m) for m,_,_ in RUNGS}
C[CTRL[0]]=ext_counts(CTRL[0])
common=set.intersection(*[set(C[m].keys()) for m in C]); common=sorted(common)
print(f"common proteins = {len(common)}")
def f1(a):
    tp,fn,fp=a.sum(0); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
def arr(model,tol): return np.array([C[model][k][tol] for k in common],float)
def F1(model,tol): return f1(arr(model,tol))
def marg(model,tol,B=2000):
    a=arr(model,tol); n=len(a); pt=f1(a)
    bs=np.array([f1(a[s]) for s in (rng.integers(0,n,n) for _ in range(B))]); return pt,*np.percentile(bs,[2.5,97.5])
def paired(mh,mr,tol=3,B=3000):
    ah=arr(mh,tol); ar=arr(mr,tol); n=len(ah)
    bs=np.array([f1(ah[s])-f1(ar[s]) for s in (rng.integers(0,n,n) for _ in range(B))]); return np.percentile(bs,[2.5,97.5])

keys=[m for m,_,_ in RUNGS]
f1pt={}; f1ci={}
for m in keys+[CTRL[0]]:
    pt,lo,hi=marg(m,3); f1pt[m]=pt; f1ci[m]=(lo,hi)
print("F1@±3 on common set:"); [print(f"  {nm:28s} {f1pt[m]:.4f}") for m,nm,_ in RUNGS]
print(f"  {CTRL[1]:28s} {f1pt[CTRL[0]]:.4f}")
deltas=[None]+[f1pt[keys[i]]-f1pt[keys[i-1]] for i in range(1,len(keys))]
dci=[None]+[paired(keys[i],keys[i-1]) for i in range(1,len(keys))]
# per-fold robustness: a pooled-significant Δ can still be driven by one fold only
F2=[k for k in common if k[0]==2]; F5=[k for k in common if k[0]==5]
def f1sub(model,ks,tol=3): return f1(np.array([C[model][x][tol] for x in ks],float))
dfold=[None]+[(f1sub(keys[i],F2)-f1sub(keys[i-1],F2),f1sub(keys[i],F5)-f1sub(keys[i-1],F5)) for i in range(1,len(keys))]
def mark(i):
    lo=dci[i][0]; d2,d5=dfold[i]
    if lo<=0: return "≈0"
    return "✓✓" if (d2>0 and d5>0) else "✓*"

# ---------- FIGURE 1: climbing F1 ladder (incremental frames + final) ----------
def draw_ladder(upto):
    # upto 1..5 = first `upto` rungs; upto==6 adds the nocompress control as the 6th step
    nr=min(upto,5); show_ctrl=upto>=6
    fig,ax=plt.subplots(figsize=(10.0,5.6))
    ax.set_xlim(-0.6,5.7); ax.set_ylim(0.50,0.74)
    ax.set_ylabel("F1")
    xt=list(range(nr)); xtl=list(DISP[:nr])  # x-labels appear together with their bars
    if show_ctrl: xt=xt+[4.95]; xtl=xtl+["адаптер\n(256→2560)"]
    ax.set_xticks(xt); ax.set_xticklabels(xtl,rotation=0,ha="center",fontsize=8.8)
    ax.grid(axis="x",alpha=0); ax.tick_params(length=0)
    xs=list(range(nr)); ys=[f1pt[keys[i]] for i in xs]
    if nr>=2: ax.plot(xs,ys,"-",color="#c9c9c9",lw=2,zorder=1)
    for i in xs:
        m,nm,col=RUNGS[i]; lo,hi=f1ci[m]
        ax.errorbar(i,f1pt[m],yerr=[[f1pt[m]-lo],[hi-f1pt[m]]],fmt="o",color=col,ms=12,mec="white",
                    mew=1.6,ecolor=col,elinewidth=2,capsize=4,zorder=3)
        if not PRES:
            ax.annotate(f"{f1pt[m]:.3f}",(i,hi),textcoords="offset points",xytext=(0,8),ha="center",
                        fontsize=9,color=MUTE,weight="bold")
        if i>0 and deltas[i] is not None:
            mid=(f1pt[keys[i-1]]+f1pt[keys[i]])/2
            ax.annotate(f"Δ={deltas[i]:+.03f}",(i-0.5,mid),textcoords="offset points",
                        xytext=(0,-26),ha="center",fontsize=8.8,color=col,weight="bold")
    # сжатие control (6th step): nocompress (2560) vs +3Di (256) differ ONLY in compression → ~equal.
    if show_ctrl:
        m=CTRL[0]; gx,gy=4,f1pt[keys[4]]; cx,cy=4.95,f1pt[m]; clo,chi=f1ci[m]
        ax.plot([gx,cx],[gy,cy],"--",color=CTRL[2],lw=1.0,alpha=0.7,zorder=2)
        ax.errorbar(cx,cy,yerr=[[cy-clo],[chi-cy]],fmt="none",ecolor=CTRL[2],elinewidth=2,capsize=4,zorder=4)
        ax.plot(cx,cy,"o",mfc="white",mec=CTRL[2],mew=2.2,ms=12,zorder=5)
        ax.annotate(f"{cy:.3f}",(cx,chi),textcoords="offset points",xytext=(0,7),ha="center",
                    va="bottom",fontsize=8.5,color="#3f7a60",weight="bold")  # above the CI cap, no overlap
        dcomp=float(f"{cy:.3f}")-float(f"{f1pt[keys[4]]:.3f}")  # nocompress - green, matches shown values
        ax.annotate(f"Δ={dcomp:+.03f}",(cx,clo),textcoords="offset points",xytext=(0,-7),ha="center",
                    va="top",fontsize=8.2,color="#3f7a60",weight="bold")
        if not PRES:
            ax.annotate("без сжатия\n→ сжатие 16× бесплатно",(cx,cy),
                        textcoords="offset points",xytext=(15,0),fontsize=7.4,color="#3f7a60",
                        style="italic",ha="left",va="center")
    ax.set_title(ptitle("Итеративное улучшение модели: что добавляет каждый шаг\n(исправл. метрика, объединение фолдов {2} и {5}, 95 % ДИ)"),fontsize=12.5,pad=12)
    if not PRES:
        fig.text(0.5,-0.04,"Прирости значимы в среднем по объединению фолдов; устойчив на обоих фолдах только шаг boundary-головы.",
                 ha="center",fontsize=7.6,color=MUTE)
    fig.tight_layout(); fig.savefig(OUT+f"ladder_f{upto}.png",bbox_inches="tight")
    if upto==6:
        for p in final_dirs("ladder.png"): fig.savefig(p,bbox_inches="tight")
    plt.close(fig)
for k in range(1,7): draw_ladder(k)

# ---------- FIGURE 2: tolerance pair (incremental frames + final) ----------
def draw_tol(upto):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.8))
    rdy={0:0,1:0,2:0,3:7,4:-7}  # stagger adapter/3Di retention labels so they don't collide
    np_=len(TOLPLOT); i3=TOLPLOT.index(REFTOL); last=np_-1  # ±0 position
    for i in range(min(upto,5)):
        m,nm,col=RUNGS[i]; ys=[F1(m,t) for t in TOLPLOT]; ref=F1(m,REFTOL); ret=[y/ref for y in ys]
        ax1.plot(range(np_),ys,"o-",color=col,lw=2,ms=6,mec="white",mew=1.1,label=nm)
        ax2.plot(range(np_),ret,"o-",color=col,lw=2,ms=6,mec="white",mew=1.1,label=nm)
        ax2.annotate(f"{ret[last]:.2f}",(last,ret[last]),textcoords="offset points",xytext=(8,rdy[i]),
                     fontsize=8.3,color=col,weight="bold")
    if upto>=6:  # 6th step: nocompress as a dashed curve (legend only, no right-side number — crowded)
        m=CTRL[0]; ys=[F1(m,t) for t in TOLPLOT]; ref=F1(m,REFTOL); ret=[y/ref for y in ys]
        ax1.plot(range(np_),ys,"--o",color=CTRL[2],lw=2,ms=6,mec="white",mew=1.1,label="без сжатия (256→2560)")
        ax2.plot(range(np_),ret,"--o",color=CTRL[2],lw=2,ms=6,mec="white",mew=1.1,label="без сжатия (256→2560)")
    for ax in (ax1,ax2):
        ax.set_xticks(range(np_)); ax.set_xticklabels([f"±{t}" for t in TOLPLOT]); ax.grid(axis="x",alpha=0)
        ax.set_xlabel("допуск совпадения границ"); ax.tick_params(length=0)
        ax.axvline(i3,color="#aaa",ls="--",lw=1.2,zorder=0)  # mark the ±3 (detection) level
    ax2.axhline(1.0,color="#aaa",ls="--",lw=1.0,zorder=0)    # ±3 reference (=1.0)
    ax1.set_ylim(0.28,0.74); ax1.set_ylabel("F1"); ax1.set_title(ptitle("(а) F1 при ужесточении допуска"))
    if not PRES:
        ax1.annotate("±3",(i3,0.74),textcoords="offset points",xytext=(4,-2),ha="left",va="top",fontsize=8,color="#999")
    ax1.legend(loc="lower left",fontsize=8.3)
    ax2.set_ylim(0.45,1.12); ax2.set_ylabel("доля F1 относительно ±3  (F1@±tol / F1@±3)")
    ax2.set_title(ptitle("(б) Доля сохранённого F1 (точность границ)"))
    fig.suptitle(ptitle("Точность границ по шагам: каждый шаг добавляет одну ломаную (объединение фолдов {2} и {5})"),
                 fontsize=13,weight="bold",color=INK,y=1.02)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT+f"ladder_tol_f{upto}.png",bbox_inches="tight")
    if upto==6:
        for p in final_dirs("ladder_tol.png"): fig.savefig(p,bbox_inches="tight")
    plt.close(fig)
for k in range(1,7): draw_tol(k)
print(f"wrote {OUT} : ladder.png + ladder_tol.png + frames f1..f6")
