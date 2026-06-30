"""Report similarity figure (replaces fig:similarity): 3 models with 95% CI, ALL segments.
3 models = baseline (gray) + the two top: zeroctrl/proj (purple) & gated_boundary/3di (green)
-- the green & purple of the report color scheme. Panel (b) annotates the per-model rise (Δ).
(a) recall vs genus TRAIN-abundance bins (representation) -> flat, rho~0
(b) recall vs SEGMENT identity-to-train bins (similarity)  -> rises (Δ labeled)
Reads seg_matched_2026.csv + seg_matched_zeroctrl.csv + identity_2026.csv. No GPU.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _pres import MODEL_LABELS, COLORS, PRES, title as ptitle, outdirs
rng=np.random.default_rng(42)
FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"
sm=pd.concat([pd.read_csv("analysis/similarity/seg_matched_2026.csv"),
              pd.read_csv("analysis/similarity/seg_matched_zeroctrl.csv")],ignore_index=True)
idn=pd.read_csv("analysis/similarity/identity_2026.csv")
d=sm.merge(idn,on=["task","seq"],how="left")
MODELS=["baseline_esm2","esmc6b_3di_zeroctrl","esmc6b_3di_gated_boundary"]
LAB={m:MODEL_LABELS[m] for m in MODELS}
COL={m:COLORS[m] for m in MODELS}
d=d[d.model.isin(MODELS)]
common=set.intersection(*[set(d[d.model==m].protein) for m in MODELS]); d=d[d.protein.isin(common)]
# genus train-abundance (all types, train folds 0,3,4,6)
def parse(s): return re.findall(r"\((\d+)-(\d+)\)",s) if isinstance(s,str) else []
trf=pd.read_csv("data/uniprot_2026/labeled_sequences.csv").merge(
    pd.read_csv("data/uniprot_2026/graphpart_assignments.csv").rename(columns={"AC":"protein_id","cluster":"fold"}),on="protein_id")
trf["genus"]=trf.organism.fillna("?").str.split().str[0]; tcnt={}
for _,r in trf[trf.fold.isin([0,3,4,6])].iterrows():
    n=len(parse(r["coordinates"]))+len(parse(r["propeptide_coordinates"]))
    if n: tcnt[r["genus"]]=tcnt.get(r["genus"],0)+n
d["train_count"]=d.genus.map(lambda g:tcnt.get(g,0))
abins=[0,5,20,100,1e9]; albl=["1–5","6–20","21–100","100+"]
d["ab"]=pd.cut(d.train_count,bins=abins,labels=albl,right=True)
ibins=[0,0.30,0.40,0.50,0.60,0.70,1.0001]; ilbl=["<0.30","0.30–0.40","0.40–0.50","0.50–0.60","0.60–0.70","≥0.70"]
d["ib"]=pd.cut(d.max_identity_to_train,bins=ibins,labels=ilbl,right=False)
def boot(x,B=2000):
    x=np.asarray(x,float); n=len(x); pt=x.mean()
    bs=np.array([x[rng.integers(0,n,n)].mean() for _ in range(B)]); return pt,*np.percentile(bs,[2.5,97.5]),n
def spearman(x,y):
    rx=pd.Series(x).rank().to_numpy(); ry=pd.Series(y).rank().to_numpy(); return np.corrcoef(rx,ry)[0,1]
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#ececec","xtick.color":"#555","ytick.color":"#555","legend.frameon":False,"legend.fontsize":9})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.2))
rhos={}
for m in MODELS:
    s=d[d.model==m].copy()
    g=s.groupby("genus").agg(recall=("matched","mean"),n=("matched","size"),tc=("train_count","first")).reset_index()
    g=g[(g.n>=5)&(g.tc>=1)]; rhos[m]=spearman(g.tc,g.recall)
# panel (a) representation
for m in MODELS:
    s=d[d.model==m]; pts=[];los=[];his=[];xs=[]
    for bi,b in enumerate(albl):
        cell=s[s.ab==b].matched
        if len(cell)>=20: mm=boot(cell); pts.append(mm[0]);los.append(mm[1]);his.append(mm[2]);xs.append(bi)
    ax1.plot(xs,pts,"o-",color=COL[m],lw=2.2,ms=7,mec="white",mew=1.2,label=LAB[m])
    ax1.fill_between(xs,los,his,color=COL[m],alpha=0.14,linewidth=0)
if not PRES:
    for bi,b in enumerate(albl):
        nn=len(d[(d.model==MODELS[0])&(d.ab==b)]); ax1.text(bi,0.02,f"n={nn}",ha="center",fontsize=7.5,color="#999")
ax1.set_xticks(range(len(albl))); ax1.set_xticklabels(albl,fontsize=9.5)
ax1.set_xlabel("число обучающих сегментов рода"); ax1.set_ylabel("полнота (доля найденных сегментов)")
ax1.set_ylim(0,1.02); ax1.legend(loc="upper left"); ax1.grid(axis="x",alpha=0); ax1.tick_params(length=0)
ax1.set_title(ptitle("(а) Полнота против представленности рода")+("" if PRES else "\n"+r"$\rho_{Спирмена}$: "+", ".join(f"{rhos[m]:+.2f}" for m in MODELS)))
# panel (b) similarity + per-model Δ (rise across identity)
dy={"baseline_esm2":0,"esmc6b_3di_zeroctrl":11,"esmc6b_3di_gated_boundary":-11}
for m in MODELS:
    s=d[d.model==m]; pts=[];los=[];his=[];xs=[]
    for bi,b in enumerate(ilbl):
        cell=s[s.ib==b].matched
        if len(cell)>=20: mm=boot(cell); pts.append(mm[0]);los.append(mm[1]);his.append(mm[2]);xs.append(bi)
    ax2.plot(xs,pts,"o-",color=COL[m],lw=2.2,ms=7,mec="white",mew=1.2,label=LAB[m])
    ax2.fill_between(xs,los,his,color=COL[m],alpha=0.14,linewidth=0)
    if len(pts)>=2:
        dlt=pts[-1]-pts[0]
        ax2.annotate(f"Δ = {dlt:+.02f}",(xs[-1],pts[-1]),textcoords="offset points",xytext=(8,dy[m]),
                     fontsize=9,color=COL[m],weight="bold",ha="left",va="center")
if not PRES:
    for bi,b in enumerate(ilbl):
        nn=len(d[(d.model==MODELS[0])&(d.ib==b)]); ax2.text(bi,0.02,f"n={nn}",ha="center",fontsize=7.2,color="#999")
ax2.set_xticks(range(len(ilbl))); ax2.set_xticklabels(ilbl,rotation=15,fontsize=9)
ax2.set_xlabel("макс. идентичность сегмента к обучающим"); ax2.set_ylim(0,1.02); ax2.set_xlim(-0.4,len(ilbl)+0.3)
ax2.grid(axis="x",alpha=0); ax2.tick_params(length=0)
ax2.set_title(ptitle("(б) Полнота против похожести сегмента к обучающим\n(Δ — прирост от наименее к наиболее похожим)"))
fig.suptitle(ptitle("Что определяет полноту: представленность рода (а) против похожести к обучающим (б)\n"
             "(объединение фолдов {2} и {5}, все сегменты, 95 % ДИ)"),
             fontsize=13,weight="bold",color="#1a1a1a",y=1.05)
if not PRES:
    fig.text(0.5,-0.03,"идентичность измеряется между отдельными сегментами, а не между целыми белками "
             "(гомологичное разбиение 30 % — на уровне белков)",ha="center",fontsize=8.2,color="#888")
fig.tight_layout(rect=[0,0.01,1,0.95])
for d_ in outdirs(): fig.savefig(d_+"similarity.png",bbox_inches="tight")
print("wrote similarity.png (3 models, all segments)", "(PRES)" if PRES else "")
print("rho:",{m:round(rhos[m],2) for m in MODELS})
print("abundance-bin n:",{b:int(len(d[(d.model==MODELS[0])&(d.ab==b)])) for b in albl})
print("identity-bin n:",{b:int(len(d[(d.model==MODELS[0])&(d.ib==b)])) for b in ilbl})
