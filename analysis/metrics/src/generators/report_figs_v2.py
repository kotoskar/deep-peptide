"""Rebuild/fix report figures in clean Russian style (no GPU; reads existing CSVs).
Produces: scoreboard, trades (ΔP/ΔR by type pooled), bylength (F1/P/R), genus_recall,
data_distributions (genus common names), interaction (footnote fixed), datascale_curve (fixed).
Run: env/bin/python this.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, os.path.dirname(__file__))
import re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import MODEL_LABELS, COLORS, PRES, title as ptitle, outdirs
FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"; FOLDS=[2,5]
rng=np.random.default_rng(42)
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12.5,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","xtick.labelsize":9.5,"ytick.labelsize":10,
 "legend.frameon":False,"legend.fontsize":9.5,"figure.facecolor":"white"})
INK="#1a1a1a"; MUTE="#8a9099"
T=pd.read_csv(FIG.replace("figures/","")+"clean_tol_true.csv")
P=pd.read_csv(FIG.replace("figures/","")+"clean_tol_pred.csv")

def keyset(m):
    t=set((f,p) for f,p in zip(T[T.model==m].fold,T[T.model==m].protein) if f in FOLDS)
    pr=set((f,p) for f,p in zip(P[P.model==m].fold,P[P.model==m].protein) if f in FOLDS)
    return t|pr
def common(models):
    s=None
    for m in models: s=keyset(m) if s is None else s&keyset(m)
    return s
def f1pr(tp,fn,fp):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return (2*p*r/(p+r) if p+r else 0),p,r

LAD_M=[("baseline_esm2","ladder"),("esmc_600m","ladder"),("esmc_6b","ladder"),
       ("esmc6b_boundary","ladder"),("esmc6b_3di_gated_boundary","ladder"),
       ("esmc6b_3di_zeroctrl","control"),("esmc6b_3di_nocompress","control")]
LAD=[(m,MODEL_LABELS[m],COLORS[m],role) for m,role in LAD_M]
cprot=common([m for m,_,_,_ in LAD])

def model_pp(m, which):  # per-(fold,protein) arrays of tp/fn/fp at tol=3 on common set
    t=T[(T.model==m)&(T.fold.isin(FOLDS))]; q=P[(P.model==m)&(P.fold.isin(FOLDS))]
    if which!="all": t=t[t.task==which]; q=q[q.task==which]
    t=t[[(f,p) in cprot for f,p in zip(t.fold,t.protein)]]
    q=q[[(f,p) in cprot for f,p in zip(q.fold,q.protein)]]
    tg=t.groupby(["fold","protein"]).m3.agg(tp="sum",ntrue="size")
    pg=q.groupby(["fold","protein"]).m3.agg(mp="sum",npred="size")
    d=tg.join(pg,how="outer").fillna(0.0); d["fn"]=d.ntrue-d.tp; d["fp"]=d.npred-d.mp
    return d[["tp","fn","fp"]].to_numpy()
def metric(arr,kind):
    tp,fn,fp=arr.sum(0); f,p,r=f1pr(tp,fn,fp); return {"f1":f,"p":p,"r":r}[kind]
def marg(m,kind,B=2000):
    arr=model_pp(m,"all"); n=len(arr); pt=metric(arr,kind)
    bs=np.array([metric(arr[rng.integers(0,n,n)],kind) for _ in range(B)])
    return pt,*np.percentile(bs,[2.5,97.5])

# ===== 1. SCOREBOARD (Russian) =====
fig,axs=plt.subplots(1,3,figsize=(14.6,4.7),sharey=False)
f1order=sorted(range(len(LAD)),key=lambda i: marg(LAD[i][0],"f1")[0])
ypos={i:yi for yi,i in enumerate(f1order)}
oi=[i for i,t in enumerate(LAD) if t[0]=="esmc6b_boundary"][0]   # orange (no gated)
for ax,(kind,lab) in zip(axs,[("f1","F1"),("p","Точность (precision)"),("r","Полнота (recall)")]):
    vals=[(nm,col,role,*marg(m,kind)) for m,nm,col,role in LAD]
    for i,yi in ypos.items():
        nm,col,role,pt,lo,hi=vals[i]
        ax.plot([lo,hi],[yi,yi],color=col,lw=3,alpha=0.40,solid_capstyle="round")
        if role=="control":
            ax.plot(pt,yi,"o",mfc="white",mec=col,mew=2.0,ms=9,zorder=3)   # hollow = control
        else:
            ax.plot(pt,yi,"o",color=col,ms=9,mec="white",mew=1.2,zorder=3) # filled = ladder
        if not PRES:
            ax.text(hi+0.006,yi,f"{pt:.3f}",va="center",ha="left",fontsize=8.5,color=MUTE)
    # dashed separator + honest note: the step INTO the gated cluster is not isolated
    ax.axhline(ypos[oi]+0.5,color="#c4c4c4",ls="--",lw=1.0,alpha=0.8,zorder=0)
    ax.set_yticks(range(len(LAD))); ax.set_yticklabels([vals[i][0] for i in f1order] if kind=="f1" else [],fontsize=8.0)
    ax.set_title(ptitle(lab)); ax.set_xlim(0.45,0.82); ax.grid(axis="y",alpha=0); ax.tick_params(length=0)
    if kind=="f1" and not PRES:
        ax.text(0.452,ypos[oi]+0.55,"↑ gated-адаптер: +0.022 (изолировано абляцией)\nзначимо в среднем, но прирост в осн. на фолде 5",
                fontsize=7.0,color="#8a6a2a",va="bottom",ha="left",style="italic")
fig.suptitle(ptitle("Точность, полнота и F1 моделей на объединении отложенных фолдов {2} и {5}"),
             fontsize=14,weight="bold",color=INK,y=1.02)
if not PRES:
    fig.text(0.5,0.945,f"допуск ±3, исправленная метрика, 95 % ДИ · n={len(cprot)} белков · отсортировано по F1  ·  "
             "полые ○ — контроли gated-модели (3Di выкл / без сжатия): обе ложатся на полную модель → 3Di и сжатие нейтральны",
             ha="center",fontsize=8.0,color=MUTE)
fig.tight_layout(rect=[0,0,1,0.92])
for d in outdirs(): fig.savefig(d+"scoreboard.png",bbox_inches="tight")
print("scoreboard.png", "(PRES)" if PRES else "")

# presentation uses effects_slopes (3Di+bond) and the dedicated bylength figure;
# the stale trades/bylength below are report-only and overwritten by their own scripts.
if PRES:
    raise SystemExit

# ===== 2. TRADES (3Di) ΔP/ΔR by type, pooled, Russian =====
WIN,ZERO="esmc6b_3di_gated_boundary","esmc6b_3di_zeroctrl"
cz=common([WIN,ZERO])
def paired_typed(kind,task,B=3000):
    def pp(m):
        t=T[(T.model==m)&(T.fold.isin(FOLDS))&(T.task==task)]; q=P[(P.model==m)&(P.fold.isin(FOLDS))&(P.task==task)]
        t=t[[(f,p) in cz for f,p in zip(t.fold,t.protein)]]; q=q[[(f,p) in cz for f,p in zip(q.fold,q.protein)]]
        tg=t.groupby(["fold","protein"]).m3.agg(tp="sum",ntrue="size"); pg=q.groupby(["fold","protein"]).m3.agg(mp="sum",npred="size")
        d=tg.join(pg,how="outer").fillna(0.0); d["fn"]=d.ntrue-d.tp; d["fp"]=d.npred-d.mp; return d[["tp","fn","fp"]]
    A=pp(WIN); B_=pp(ZERO); idx=A.index.intersection(B_.index)
    A=A.loc[idx].to_numpy(); Bm=B_.loc[idx].to_numpy(); n=len(idx)
    pt=metric(A,kind)-metric(Bm,kind)
    bs=np.array([metric(A[s],kind)-metric(Bm[s],kind) for s in (rng.integers(0,n,n) for _ in range(B))])
    return pt,*np.percentile(bs,[2.5,97.5])
fig,ax=plt.subplots(figsize=(7.6,4.6))
RED="#d1495b"; BLU="#3c7bb0"; x=np.arange(2); w=0.32
for k,(kind,col,nm,dx) in enumerate([("p",RED,"Δ точность",-w/2),("r",BLU,"Δ полнота",w/2)]):
    pts=[];los=[];his=[]
    for task in ("pep","propep"):
        pt,lo,hi=paired_typed(kind,task); pts.append(pt);los.append(pt-lo);his.append(hi-pt)
    bars=ax.bar(x+dx,pts,w,color=col,label=nm,zorder=3,edgecolor="white",linewidth=1)
    ax.errorbar(x+dx,pts,yerr=[los,his],fmt="none",ecolor=INK,elinewidth=1.3,capsize=0,zorder=4)
ax.axhline(0,color="#999",lw=1); ax.set_xticks(x); ax.set_xticklabels(["пептиды","пропептиды"])
ax.set_ylabel("разница метрики   (3Di − без 3Di)"); ax.grid(axis="x",alpha=0); ax.tick_params(length=0); ax.legend(loc="lower left")
ax.set_title("Изменение точности и полноты при добавлении 3Di,\nпо типам сегментов (объединение фолдов {2} и {5})",fontsize=12.5,pad=11)
fig.tight_layout(); fig.savefig(FIG+"trades.png",bbox_inches="tight"); fig.savefig(OUT+"trades.png",bbox_inches="tight")
print("trades.png")

# ===== 3. F1/P/R BY LENGTH =====
LB=["5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-50"]
def lbin(L):
    e=[5,10,15,20,25,30,35,40,45,51]
    for i in range(len(e)-1):
        if e[i]<=L<e[i+1]: return f"{e[i]}-{e[i+1]-1}"
    return None
def bylen(m):
    t=T[(T.model==m)&(T.fold.isin(FOLDS))].copy(); q=P[(P.model==m)&(P.fold.isin(FOLDS))].copy()
    t=t[[(f,p) in cprot for f,p in zip(t.fold,t.protein)]]; q=q[[(f,p) in cprot for f,p in zip(q.fold,q.protein)]]
    t["lb"]=t.length.map(lbin); q["lb"]=q.plen.map(lbin); res={}
    for k in LB:
        tk=t[t.lb==k]; qk=q[q.lb==k]
        if len(tk)<25: continue
        res[k]=f1pr(tk.m3.sum(),(tk.m3==0).sum(),(qk.m3==0).sum())+(len(tk),)
    return res
fig,axs=plt.subplots(1,3,figsize=(15.5,4.7),sharey=True)
keys_present=[k for k in LB if k in bylen(LAD[0][0])]
for ax,(kind,lab,mi) in zip(axs,[("F1","F1",0),("P","Точность",1),("R","Полнота",2)]):
    for m,nm,col,role in LAD:
        if role!="ladder": continue
        r=bylen(m); ks=[k for k in LB if k in r]; xs=np.arange(len(ks))
        ax.plot(xs,[r[k][mi] for k in ks],"o-",color=col,label=nm.split("  (")[0],ms=4.5,lw=1.7)
    ax.set_xticks(range(len(keys_present))); ax.set_xticklabels(keys_present,rotation=45,fontsize=8)
    ax.set_title(lab); ax.set_xlabel("длина пептида (аминокислот)"); ax.set_ylim(0,1); ax.grid(axis="x",alpha=0)
axs[0].set_ylabel("значение метрики"); axs[0].legend(loc="lower center",fontsize=8)
fig.suptitle("F1, точность и полнота в зависимости от длины пептида (объединение фолдов {2} и {5})",fontsize=14,weight="bold",color=INK,y=1.02)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG+"bylength.png",bbox_inches="tight"); fig.savefig(OUT+"bylength.png",bbox_inches="tight")
print("bylength.png")
print("DONE part A")
