import json, math, statistics as st
d=json.load(open('/home/oskar/work/DeepPeptide/analysis/metrics/segment_quality_cv_with_esmc.json'))
M=d['models']
def po(m,g,k='d_iou_mean'): return M[m]['paired_vs_baseline'][g][k]['per_outer']

def t1(v):
    m=st.mean(v); s=st.stdev(v); t=m/(s/math.sqrt(len(v)))
    return m,s,t

print('one-sample t vs 0 over the 5 outer folds (df=4; |t|>2.776 => p<0.05 two-sided)')
for g in ['overlap','iou50','tol3']:
    print(' gate',g)
    for m in ['5cv_esmc6b_plain','5cv_esm2_boundary','5cv_esm2_adapter_only','5cv_esm2_full',
              '5cv_esmc6b_boundary','5cv_esmc6b_full']:
        mm,s,t=t1(po(m,g))
        print('   %-24s %+0.6f +- %0.6f  t=%+5.2f  %s' % (m,mm,s,t,'SIG' if abs(t)>2.776 else 'ns'))

print('\npaired-by-fold contrast: ESM-C base vs ESM-2+head (both are deltas on the same base)')
a=po('5cv_esmc6b_plain','iou50'); b=po('5cv_esm2_boundary','iou50')
diff=[x-y for x,y in zip(a,b)]
mm,s,t=t1(diff)
print('   d(ESM-C base) - d(ESM-2+head) = %+0.6f +- %0.6f  t=%+5.2f  %s  per-fold %s'
      % (mm,s,t,'SIG' if abs(t)>2.776 else 'ns', ['%+.4f'%x for x in diff]))

print('\nadditivity at iou50: capacity-alone + additions-on-ESM2  vs  observed ESM-C+both')
cap=po('5cv_esmc6b_plain','iou50'); add=po('5cv_esm2_full','iou50'); obs=po('5cv_esmc6b_full','iou50')
inter=[o-(c+a) for o,c,a in zip(obs,cap,add)]
mm,s,t=t1(inter)
print('   predicted additive mean %+0.6f, observed %+0.6f' % (st.mean(cap)+st.mean(add), st.mean(obs)))
print('   interaction = %+0.6f +- %0.6f  t=%+5.2f  %s  per-fold %s'
      % (mm,s,t,'SIG' if abs(t)>2.776 else 'ns', ['%+.4f'%x for x in inter]))

print('\nfrac_tighter / frac_looser at iou50 (median delta is 0.0 everywhere)')
for m in ['5cv_esmc6b_plain','5cv_esm2_boundary','5cv_esmc6b_boundary','5cv_esm2_full','5cv_esmc6b_full']:
    b=M[m]['paired_vs_baseline']['iou50']
    print('   %-24s tighter %.3f  looser %.3f  unchanged %.3f  median %.4f' %
          (m,b['frac_tighter']['mean'],b['frac_looser']['mean'],
           1-b['frac_tighter']['mean']-b['frac_looser']['mean'],b['d_iou_median']['mean']))
