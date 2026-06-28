"""Max identity-to-train for each held-out segment in seg_matched_2026.csv.
Train reference = FULL folds {0,3,4,6} (same for all models). Per type, needleall.
Out: analysis/similarity/identity_2026.csv  (type, seq, max_identity_to_train)
"""
import os, sys, warnings, tempfile, subprocess; warnings.filterwarnings("ignore"); sys.path.insert(0,os.getcwd())
import re, pandas as pd
from pathlib import Path
TRAIN={0,3,4,6}
def parse(s): return [(int(a),int(b)) for a,b in re.findall(r"\((\d+)-(\d+)\)",s)] if isinstance(s,str) else []
lab=pd.read_csv("data/uniprot_2026/labeled_sequences.csv")
asg=pd.read_csv("data/uniprot_2026/graphpart_assignments.csv").rename(columns={"AC":"protein_id","cluster":"fold"})
df=lab.merge(asg,on="protein_id")
# train unique seqs per type (merged-overlap, len 5..50)
def merge(segs):
    segs=sorted(segs); out=[]
    for s,e in segs:
        if out and s<=out[-1][1]+1: out[-1]=(out[-1][0],max(out[-1][1],e))
        else: out.append((s,e))
    return out
train={"pep":set(),"propep":set()}
for _,r in df[df.fold.isin(TRAIN)].iterrows():
    seq=r["sequence"]
    for col,typ in [("coordinates","pep"),("propeptide_coordinates","propep")]:
        for s,e in merge(parse(r[col])):
            p=seq[s-1:e]
            if 5<=len(p)<=50: train[typ].add(p)
print("train unique:",{k:len(v) for k,v in train.items()})
# held-out seqs from seg_matched_2026 (union over models = same set)
sm=pd.read_csv("analysis/similarity/seg_matched_2026.csv")
def write_fa(seqs,path):
    with open(path,"w") as f:
        for i,s in enumerate(seqs): f.write(f">q{i}\n{s}\n")
    return {f"q{i}":s for i,s in enumerate(seqs)}
def needle_maxid(qseqs,dseqs,wd):
    qmap=write_fa(qseqs,wd/"q.fa"); dmap=write_fa(dseqs,wd/"db.fa")
    best={q:-1.0 for q in qmap}
    proc=subprocess.Popen(["needleall","-auto","-asequence",str(wd/"q.fa"),"-bsequence",str(wd/"db.fa"),
        "-gapopen","10","-gapextend","0.5","-aformat3","pair","-errfile",str(wd/"e.txt"),"-outfile","stdout"],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    cq=None
    for line in proc.stdout:
        if line.startswith("# 1:"): cq=line.split(":",1)[1].strip()
        elif line.startswith("# Identity:"):
            x,y=line.split(":",1)[1].strip().split("(")[0].strip().split("/")
            idn=int(x)/int(y) if int(y) else 0.0
            if cq in best and idn>best[cq]: best[cq]=idn
    proc.wait()
    return {qmap[q]:v for q,v in best.items()}
rows=[]
for typ in ["pep","propep"]:
    hq=sorted(sm[sm.task==typ].seq.dropna().unique())
    with tempfile.TemporaryDirectory() as td:
        mp=needle_maxid(hq,sorted(train[typ]),Path(td))
    for s in hq: rows.append(dict(task=typ,seq=s,max_identity_to_train=mp.get(s,0.0)))
    print(f"{typ}: held-out unique={len(hq)} done")
out=pd.DataFrame(rows); out.to_csv("analysis/similarity/identity_2026.csv",index=False)
print("wrote analysis/similarity/identity_2026.csv",len(out))
