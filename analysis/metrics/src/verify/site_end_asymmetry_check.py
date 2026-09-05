import gzip, json, glob, collections, math, statistics as st
from pathlib import Path
import pandas as pd
from scipy import stats

MODELS = ['5cv_baseline_esm2_fp32','5cv_esm2_boundary','5cv_esm2_adapter_only','5cv_esm2_full',
          '5cv_esmc6b_plain','5cv_esmc6b_boundary','5cv_esmc6b_full']
seqs = pd.read_csv('data/uniprot_2026/labeled_sequences.csv', usecols=['protein_id','sequence'])
LEN = {p: len(s) for p, s in zip(seqs.protein_id, seqs.sequence)}

def load(m):
    cells={}
    for d in sorted(Path('runs',m).glob('outer*_inner*')):
        f=d/'segments.json.gz'
        if not f.exists(): continue
        o=int(d.name.split('outer')[1].split('_')[0]); i=int(d.name.split('inner')[1])
        cells[(o,i)]=json.load(gzip.open(f,'rt'))
    return cells

cs={m:load(m) for m in MODELS}
common=sorted(set.intersection(*(set(v) for v in cs.values())))
shared={}
for c in common:
    n=None
    for m in MODELS:
        g={r['name'] for r in cs[m][c]}
        n=g if n is None else (n&g)
    shared[c]=n
print('cells',len(common),'protein-cells',sum(len(v) for v in shared.values()))

def greedy(ts,ps,tol=0):
    if not ts or not ps: return 0
    cand=sorted((abs(p-t),i,j) for i,t in enumerate(ts) for j,p in enumerate(ps) if abs(p-t)<=tol)
    ut,up,tp=set(),set(),0
    for dd,i,j in cand:
        if i in ut or j in up: continue
        ut.add(i); up.add(j); tp+=1
    return tp

def f1(tp,nt,npd):
    p=tp/npd if npd else 0.0; r=tp/nt if nt else 0.0
    return 2*p*r/(p+r) if (p+r) else 0.0

def cell_scores(recs, names, mode):
    """mode: 'all' or 'interior'. returns dict of metric->value (micro in cell)"""
    tp=collections.Counter(); nt=collections.Counter(); npd=collections.Counter()
    for r in recs:
        if r['name'] not in names: continue
        L=LEN.get(r['name'])
        for task in ('peptides','propeptides'):
            for end,ix in (('start',0),('end',1)):
                T=[s[ix] for s in r[task]['true']]
                P=[s[ix] for s in r[task]['pred']]
                if mode=='interior':
                    if end=='start':
                        T=[x for x in T if x!=1]; P=[x for x in P if x!=1]
                    else:
                        if L is None: T=[];P=[]
                        else:
                            T=[x for x in T if x!=L]; P=[x for x in P if x!=L]
                nt[(task,end)]+=len(T); npd[(task,end)]+=len(P)
                tp[(task,end)]+=greedy(T,P,0)
    out={}
    for end in ('start','end'):
        a=b=c=0
        for task in ('peptides','propeptides'):
            out[f'{task}_{end}']=f1(tp[(task,end)],nt[(task,end)],npd[(task,end)])
            out[f'n_true_{task}_{end}']=nt[(task,end)]
            a+=tp[(task,end)]; b+=nt[(task,end)]; c+=npd[(task,end)]
        out[f'all_{end}']=f1(a,b,c); out[f'n_true_all_{end}']=b; out[f'n_pred_all_{end}']=c
    return out

def per_outer(model, mode):
    rows=[]
    for c in common:
        s=cell_scores(cs[model][c], shared[c], mode); s['outer']=c[0]; rows.append(s)
    df=pd.DataFrame(rows).groupby('outer').mean(numeric_only=True)
    return df

def show(mode):
    print('\n############ mode =',mode)
    P={m:per_outer(m,mode) for m in MODELS[:4]}
    base=P['5cv_baseline_esm2_fp32']
    print('  denominators (mean true sites per cell): all_start %.1f all_end %.1f  pred_start %.1f pred_end %.1f'
          %(base['n_true_all_start'].mean(),base['n_true_all_end'].mean(),
            base['n_pred_all_start'].mean(),base['n_pred_all_end'].mean()))
    for key in ['all','peptides','propeptides']:
        print(' --',key,' base start %.4f end %.4f'%(base[key+'_start'].mean(),base[key+'_end'].mean()))
        for m in MODELS[1:4]:
            ds=(P[m][key+'_start']-base[key+'_start']).tolist()
            de=(P[m][key+'_end']-base[key+'_end']).tolist()
            asym=[e-s for e,s in zip(de,ds)]
            def f(v):
                mm=st.mean(v); ss=st.stdev(v)
                p=2*stats.t.sf(abs(mm/(ss/math.sqrt(len(v)))),len(v)-1) if ss>0 else 0.0
                return '%+.4f±%.4f %d/5 p=%.3g'%(mm,ss,sum(1 for x in v if x>0),p)
            print('    %-24s start %-26s end %-26s asym %s'%(m,f(ds),f(de),f(asym)))

show('all')
show('interior')

# terminus / adjacency structure on true labels
tc=collections.Counter(); adj=collections.Counter()
for c in common:
    for r in cs['5cv_baseline_esm2_fp32'][c]:
        if r['name'] not in shared[c]: continue
        L=LEN.get(r['name'])
        pep_starts={s for s,e in r['peptides']['true']}
        pro_starts={s for s,e in r['propeptides']['true']}
        for task in ('peptides','propeptides'):
            for s,e in r[task]['true']:
                tc[(task,'n')]+=1
                if s==1: tc[(task,'start_at_Nterm')]+=1
                if L is not None and e==L: tc[(task,'end_at_Cterm')]+=1
                if task=='propeptides':
                    if (e+1) in pep_starts: adj['pro_end_touches_pep_start']+=1
                    adj['pro_end_n']+=1
                else:
                    if (s-1) in {ee for ss,ee in r['propeptides']['true']}: adj['pep_start_touches_pro_end']+=1
                    adj['pep_start_n']+=1
print('\n############ true-label structure (all 20 cells, shared set)')
for task in ('peptides','propeptides'):
    n=tc[(task,'n')]
    print('  %-12s n=%6d  start==1: %5d (%.1f%%)   end==L: %5d (%.1f%%)'
          %(task,n,tc[(task,'start_at_Nterm')],100*tc[(task,'start_at_Nterm')]/n,
            tc[(task,'end_at_Cterm')],100*tc[(task,'end_at_Cterm')]/n))
print('  propeptide end immediately precedes a peptide start: %d / %d (%.1f%%)'
      %(adj['pro_end_touches_pep_start'],adj['pro_end_n'],100*adj['pro_end_touches_pep_start']/adj['pro_end_n']))
print('  peptide start immediately follows a propeptide end:  %d / %d (%.1f%%)'
      %(adj['pep_start_touches_pro_end'],adj['pep_start_n'],100*adj['pep_start_touches_pro_end']/adj['pep_start_n']))

print('\n############ head-vs-adapter asymmetry contrast, interior filter')
for mode in ('all','interior'):
    P={m:per_outer(m,mode) for m in MODELS[:4]}
    base=P['5cv_baseline_esm2_fp32']
    A={}
    for m in ['5cv_esm2_boundary','5cv_esm2_adapter_only']:
        ds=(P[m]['all_start']-base['all_start']).tolist(); de=(P[m]['all_end']-base['all_end']).tolist()
        A[m]=[e-s for e,s in zip(de,ds)]
    diff=[x-y for x,y in zip(A['5cv_esm2_boundary'],A['5cv_esm2_adapter_only'])]
    mm=st.mean(diff); ss=st.stdev(diff)
    print('  %-9s head_asym %+.4f  adapter_asym %+.4f  paired diff %+.4f±%.4f %d/5 p=%.3g'
          %(mode,st.mean(A['5cv_esm2_boundary']),st.mean(A['5cv_esm2_adapter_only']),
            mm,ss,sum(1 for x in diff if x>0),2*stats.t.sf(abs(mm/(ss/math.sqrt(5))),4)))

print('\n############ DIRECT: head start-delta vs adapter start-delta (and end vs end), paired by fold')
for mode in ('all','interior'):
    P={m:per_outer(m,mode) for m in MODELS[:4]}
    base=P['5cv_baseline_esm2_fp32']
    D={}
    for m in ['5cv_esm2_boundary','5cv_esm2_adapter_only']:
        D[m]={'start':(P[m]['all_start']-base['all_start']).tolist(),
              'end':(P[m]['all_end']-base['all_end']).tolist()}
    for end in ('start','end'):
        h=D['5cv_esm2_boundary'][end]; a=D['5cv_esm2_adapter_only'][end]
        diff=[x-y for x,y in zip(h,a)]
        mm=st.mean(diff); ss=st.stdev(diff)
        print('  %-9s %-5s head %+.4f  adapter %+.4f  head-adapter %+.4f±%.4f  neg %d/5  p=%.3g  folds=%s'
              %(mode,end,st.mean(h),st.mean(a),mm,ss,sum(1 for x in diff if x<0),
                2*stats.t.sf(abs(mm/(ss/math.sqrt(5))),4),[round(x,4) for x in diff]))
