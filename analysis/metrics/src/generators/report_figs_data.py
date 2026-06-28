"""Report figures (clean style, no GPU): data_distributions + fold_divergence.
Style matches proto_scoreboard (despined, muted, light grid).
Run: env/bin/python this.py
"""
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG="analysis/metrics/figures/"; OUT="texs/Overleaf/figures/"
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":150,"font.family":"DejaVu Sans","font.size":11,
 "axes.titlesize":12,"axes.titleweight":"bold","axes.titlelocation":"center","axes.titlepad":9,
 "axes.labelsize":10.5,"axes.edgecolor":"#bbb","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#ececec",
 "xtick.color":"#555","ytick.color":"#555","xtick.labelsize":9.5,"ytick.labelsize":10,
 "legend.frameon":False,"legend.fontsize":9.5})
INK="#1a1a1a"; MUTE="#8a9099"
def segs(s): return [(int(a),int(b)) for a,b in re.findall(r"\((\d+)-(\d+)\)",s)] if isinstance(s,str) else []

# ---------- data_distributions: lengths (pep vs propep) + top organisms ----------
d26=pd.read_csv("data/uniprot_2026/labeled_sequences.csv")
pl=[b-a+1 for r in d26["coordinates"] for a,b in segs(r)]
rl=[b-a+1 for r in d26["propeptide_coordinates"] for a,b in segs(r)]
d26["genus"]=d26.organism.fillna("?").str.split().str[0]
gtop=d26.genus.value_counts().head(10)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.4),gridspec_kw={"width_ratios":[1.15,1]})
bins=np.arange(5,52,3)
ax1.hist(pl,bins=bins,alpha=0.7,color="#4c9be8",label=f"пептиды (n={len(pl)})",edgecolor="white",linewidth=0.6)
ax1.hist(rl,bins=bins,alpha=0.55,color="#e0913f",label=f"пропептиды (n={len(rl)})",edgecolor="white",linewidth=0.6)
ax1.set_xlabel("длина сегмента (аминокислот)"); ax1.set_ylabel("число сегментов")
ax1.set_title("(а) Распределение длин"); ax1.legend(); ax1.grid(axis="x",alpha=0)
COMMON={"Conus":"улитки-конусы","Mus":"мыши","Homo":"человек","Cyriopagopus":"пауки-птицееды",
        "Rattus":"крысы","Arabidopsis":"резуховидка","Bos":"коровы","Lycosa":"пауки-волки",
        "Saccharomyces":"дрожжи","Drosophila":"дрозофилы","Gallus":"куры","Sus":"свиньи"}
y=np.arange(len(gtop))[::-1]
ax2.barh(y,gtop.values,color="#7a8aa0",edgecolor="white",linewidth=0.8)
labels=[f"{g}\n({COMMON.get(g,'?')})" for g in gtop.index]
ax2.set_yticks(y); ax2.set_yticklabels(labels,fontsize=8.5)
for yi,v in zip(y,gtop.values): ax2.text(v+5,yi,str(v),va="center",fontsize=8.5,color=MUTE)
ax2.set_xlabel("число белков"); ax2.set_title("(б) Самые частые роды организмов"); ax2.grid(axis="y",alpha=0)
fig.suptitle("Распределение длин сегментов и родов организмов в датасете 2026 года",fontsize=14,weight="bold",color=INK,y=1.02)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG+"data_distributions.png",bbox_inches="tight"); fig.savefig(OUT+"data_distributions.png",bbox_inches="tight"); print("data_distributions done")

# ---------- fold_divergence: length profile per fold + L1 to global ----------
asg=pd.read_csv("data/uniprot_2026/graphpart_assignments.csv").rename(columns={"AC":"protein_id","cluster":"fold"})
df=d26.merge(asg,on="protein_id")
LBINS=[5,10,15,20,25,30,35,40,45,51]; keys=[f"{LBINS[i]}-{LBINS[i+1]-1}" for i in range(len(LBINS)-1)]
def lbin(L):
    for i in range(len(LBINS)-1):
        if LBINS[i]<=L<LBINS[i+1]: return keys[i]
    return None
rows=[]
for _,r in df.iterrows():
    for col in ("coordinates","propeptide_coordinates"):
        for a,b in segs(r.get(col)):
            lb=lbin(b-a+1)
            if lb: rows.append((int(r["fold"]),lb))
S=pd.DataFrame(rows,columns=["fold","lb"])
piv=S.groupby(["fold","lb"]).size().unstack(fill_value=0); piv=piv.div(piv.sum(1),axis=0)[keys]
glob=(S.groupby("lb").size()/len(S))[keys]
L1={f:float(np.abs(piv.loc[f].values-glob.values).sum()) for f in piv.index}

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.4),gridspec_kw={"width_ratios":[1.3,1]})
cmap=plt.cm.viridis(np.linspace(0.15,0.85,len(piv.index)))
x=np.arange(len(keys))
ax1.plot(x,glob.values*100,"o-",color=INK,lw=2.6,ms=6,label="весь датасет",zorder=5)
for i,f in enumerate(piv.index):
    ax1.plot(x,piv.loc[f].values*100,"-",color=cmap[i],lw=1.3,alpha=0.8,label=f"фолд {f}")
ax1.set_xticks(x); ax1.set_xticklabels(keys,rotation=45,fontsize=8); ax1.set_xlabel("длина сегмента")
ax1.set_ylabel("доля сегментов, %"); ax1.set_title("(а) Профиль длин по фолдам"); ax1.legend(ncol=2,fontsize=8); ax1.grid(axis="x",alpha=0)
order=sorted(L1,key=lambda f:L1[f]); yy=np.arange(len(order))[::-1]
cols=["#4ca37a" if f==order[0] else "#c25a5a" if L1[f]>0.23 else "#7a8aa0" for f in order]
ax2.barh(yy,[L1[f] for f in order],color=cols,edgecolor="white",linewidth=0.8)
ax2.set_yticks(yy); ax2.set_yticklabels([f"фолд {f}" for f in order])
for yi,f in zip(yy,order): ax2.text(L1[f]+0.004,yi,f"{L1[f]:.2f}",va="center",fontsize=8.5,color=MUTE)
ax2.set_xlabel("отклонение от общего профиля (L1)"); ax2.set_title("(б) Отклонение профиля длин фолда от общего"); ax2.grid(axis="y",alpha=0)
fig.suptitle("Распределение длин сегментов по фолдам и его отклонение от общего по датасету",fontsize=14,weight="bold",color=INK,y=1.02)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG+"fold_divergence.png",bbox_inches="tight"); print("fold_divergence done")
print("L1:",{f:round(v,3) for f,v in sorted(L1.items())})
