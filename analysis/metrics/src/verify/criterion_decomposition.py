import gzip, json, sys, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'/home/oskar/work/DeepPeptide')
from analysis.errors.src.error_analysis import _group

MODELS7 = ['5cv_baseline_esm2_fp32','5cv_esm2_boundary','5cv_esm2_adapter_only',
           '5cv_esm2_full','5cv_esmc6b_plain','5cv_esmc6b_boundary','5cv_esmc6b_full']
TASKS=('peptides','propeptides')

def load(model):
    cells={}
    for d in sorted(Path('runs',model).glob('outer*_inner*')):
        f=d/'segments.json.gz'
        if not f.exists(): continue
        o=int(d.name.split('outer')[1].split('_')[0]); i=int(d.name.split('inner')[1])
        with gzip.open(f,'rt') as fh: cells[(o,i)]=json.load(fh)
    return cells

def iou(a,b):
    inter=min(a[1],b[1])-max(a[0],b[0])+1
    if inter<=0: return 0.0
    return inter/((a[1]-a[0]+1)+(b[1]-b[0]+1)-inter)

def err(t,p): return max(abs(p[0]-t[0]),abs(p[1]-t[1]))

def greedy(true,pred,score,better_is_high):
    """one-to-one greedy; returns list of (i,j)"""
    cand=[]
    for i,t in enumerate(true):
        for j,p in enumerate(pred):
            v=score(t,p)
            cand.append((v,i,j))
    cand.sort(key=lambda x:((-x[0] if better_is_high else x[0]),x[1],x[2]))
    ut,up,out=set(),set(),[]
    for v,i,j in cand:
        if better_is_high and v<=0: continue
        if i in ut or j in up: continue
        ut.add(i); up.add(j); out.append((i,j,v))
    return out

def grouped_counts(true,pred,accept):
    """accept(t,p)->bool ; grouped-true matcher (the paper's corrected matcher generalised)"""
    if len(true)==0: return 0,0,len(pred)
    groups=_group(true)
    pm=[False]*len(pred)
    tp=fn=0
    for g in groups:
        m=False
        for t in g:
            for pi,p in enumerate(pred):
                if accept(t,p):
                    m=True; pm[pi]=True
        tp+=m; fn+=(not m)
    return tp,fn,sum(not x for x in pm)

def onetoone_counts(true,pred,score,better_is_high,accept):
    if not true or not pred:
        return 0,len(true),len(pred)
    ms=greedy(true,pred,score,better_is_high)
    tp=sum(1 for i,j,v in ms if accept(true[i],pred[j]))
    return tp,len(true)-tp,len(pred)-tp

def prf(tp,fn,fp):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    return p,r,(2*p*r/(p+r) if p+r else 0.0)

ACC3   = lambda t,p: err(t,p)<=3
ACCIOU = lambda t,p: iou(t,p)>=0.5

def score_cell(recs,keep):
    if keep is not None: recs=[r for r in recs if r['name'] in keep]
    C=defaultdict(lambda:[0,0,0])
    for rec in recs:
        for task in TASKS:
            true=[tuple(map(int,x)) for x in rec[task]['true']]
            pred=[tuple(map(int,x)) for x in rec[task]['pred']]
            if not true and not pred: continue
            for key,c in (
                ('A_grouped_tol3',   grouped_counts(true,pred,ACC3)),
                ('D_grouped_iou50',  grouped_counts(true,pred,ACCIOU)),
                ('C_1to1_iou50',     onetoone_counts(true,pred,iou,True,ACCIOU)),
                ('B1_1to1iouassign_tol3', onetoone_counts(true,pred,iou,True,ACC3)),
                ('B2_1to1errassign_tol3', onetoone_counts(true,pred,err,False,ACC3)),
            ):
                for k in range(3): C[key][k]+=c[k]
    out={}
    for key,c in C.items():
        p,r,f=prf(*c)
        out[key+'_P']=p; out[key+'_R']=r; out[key+'_F1']=f
        out[key+'_ntrue']=c[0]+c[1]; out[key+'_npred']=c[0]+c[2]
    return out

def run(keep_map,label):
    per={}
    for m in ['5cv_baseline_esm2_fp32','5cv_esm2_boundary']:
        cells=CELLS[m]
        rows=[]
        for cell in sorted(keep_map):
            r=score_cell(cells[cell], keep_map[cell])
            r['outer']=cell[0]; rows.append(r)
        per[m]=pd.DataFrame(rows).groupby('outer').mean(numeric_only=True)
    b,v=per['5cv_baseline_esm2_fp32'],per['5cv_esm2_boundary']
    print(f'\n===== {label} =====')
    for key in ['A_grouped_tol3','D_grouped_iou50','C_1to1_iou50','B1_1to1iouassign_tol3','B2_1to1errassign_tol3']:
        for met in ['F1','P','R']:
            col=key+'_'+met
            d=(v[col]-b[col]).values
            print(f'{key:24s} {met:2s}  base {b[col].mean():.4f}  head {v[col].mean():.4f}  '
                  f'delta {d.mean():+.4f} +- {d.std(ddof=1):.4f}  pos {int((d>0).sum())}/5')
        print(f'{key:24s} n_true/cell base {b[key+"_ntrue"].mean():.1f}  n_pred/cell base {b[key+"_npred"].mean():.1f} head {v[key+"_npred"].mean():.1f}')
        print()

CELLS={m:load(m) for m in MODELS7}
for m in MODELS7: print(m,len(CELLS[m]),'cells')
common_cells=sorted(set.intersection(*(set(c) for c in CELLS.values())))
shared={}
for c in common_cells:
    names=None
    for m in MODELS7:
        got={r['name'] for r in CELLS[m][c]}
        names=got if names is None else names&got
    shared[c]=names
print('common protein-cells',sum(len(v) for v in shared.values()),'cells',len(common_cells))

esm2cells=sorted(set(CELLS['5cv_baseline_esm2_fp32'])&set(CELLS['5cv_esm2_boundary']))
esm2_shared={}
for c in esm2cells:
    a={r['name'] for r in CELLS['5cv_baseline_esm2_fp32'][c]}
    bb={r['name'] for r in CELLS['5cv_esm2_boundary'][c]}
    print('  cell',c,'base',len(a),'head',len(bb),'sym-diff',len(a^bb)) if a!=bb else None
    esm2_shared[c]=None  # no filter -> full native set
print('ESM-2 native protein-cells base',sum(len(CELLS['5cv_baseline_esm2_fp32'][c]) for c in esm2cells))

run(esm2_shared,'ESM-2 NATIVE protein set (what the +-3 table uses)')
run(shared,'COMMON 35,504 protein-cells (what the IoU suite uses)')
