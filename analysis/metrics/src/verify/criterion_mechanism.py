import gzip,json,sys,numpy as np,pandas as pd
from pathlib import Path
sys.path.insert(0,'/home/oskar/work/DeepPeptide')
from collections import defaultdict
TASKS=('peptides','propeptides')
def load(m):
    c={}
    for d in sorted(Path('runs',m).glob('outer*_inner*')):
        o=int(d.name.split('outer')[1].split('_')[0]);i=int(d.name.split('inner')[1])
        with gzip.open(d/'segments.json.gz','rt') as fh: c[(o,i)]=json.load(fh)
    return c
def iou(a,b):
    it=min(a[1],b[1])-max(a[0],b[0])+1
    return it/((a[1]-a[0]+1)+(b[1]-b[0]+1)-it) if it>0 else 0.0
def err(t,p): return max(abs(p[0]-t[0]),abs(p[1]-t[1]))
def greedy_iou(true,pred):
    cand=[(iou(t,p),i,j) for i,t in enumerate(true) for j,p in enumerate(pred) if iou(t,p)>0]
    cand.sort(key=lambda x:(-x[0],x[1],x[2]))
    ut,up,out=set(),set(),{}
    for v,i,j in cand:
        if i in ut or j in up: continue
        ut.add(i);up.add(j);out[i]=(j,v)
    return out
B=load('5cv_baseline_esm2_fp32');H=load('5cv_esm2_boundary')
rows=[]
for cell in sorted(set(B)&set(H)):
    bm={r['name']:r for r in B[cell]}; hm={r['name']:r for r in H[cell]}
    names=set(bm)&set(hm)
    c=defaultdict(int)
    for n in names:
        for task in TASKS:
            true=[tuple(map(int,x)) for x in bm[n][task]['true']]
            if not true: continue
            for src,rec in (('base',bm[n]),('head',hm[n])):
                pred=[tuple(map(int,x)) for x in rec[task]['pred']]
                mt=greedy_iou(true,pred)
                for i,t in enumerate(true):
                    j,vv=mt.get(i,(None,0.0))
                    e=min((err(t,p) for p in pred),default=99)
                    if vv>=0.5:
                        c[src+'_iou50']+=1
                        if e<=3: c[src+'_iou50_and_tol3']+=1
                        else:    c[src+'_iou50_not_tol3']+=1
                    if e<=3:
                        c[src+'_tol3']+=1
                        if vv<0.5: c[src+'_tol3_not_iou50']+=1
    c['outer']=cell[0]; rows.append(dict(c))
df=pd.DataFrame(rows).groupby('outer').mean(numeric_only=True)
print('per-cell counts over TRUE segments (mean of per-outer means):')
for k in sorted(df.columns):
    print(f'  {k:26s} {df[k].mean():9.1f}')
print()
print('DELTA head-base per outer:')
for k in ['iou50','iou50_and_tol3','iou50_not_tol3','tol3','tol3_not_iou50']:
    d=(df['head_'+k]-df['base_'+k]).values
    print(f'  {k:20s} {d.mean():+8.1f} +- {d.std(ddof=1):6.1f}  pos {int((d>0).sum())}/5')
