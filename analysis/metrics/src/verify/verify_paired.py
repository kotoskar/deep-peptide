import gzip, json, os, itertools, statistics as st
from collections import defaultdict

os.chdir('/home/oskar/work/DeepPeptide')
MODELS = ['5cv_baseline_esm2_fp32','5cv_esm2_boundary','5cv_esm2_adapter_only','5cv_esm2_full',
          '5cv_esmc6b_plain','5cv_esmc6b_boundary','5cv_esmc6b_full']
BASE = '5cv_baseline_esm2_fp32'
TASKS = ('peptides','propeptides')
CAP = 50

def load(model):
    cells = {}
    root = 'runs/' + model
    for d in sorted(os.listdir(root)):
        f = os.path.join(root, d, 'segments.json.gz')
        if not (d.startswith('outer') and os.path.exists(f)):
            continue
        o = int(d.split('outer')[1].split('_')[0]); i = int(d.split('inner')[1])
        cells[(o,i)] = json.load(gzip.open(f,'rt'))
    return cells

def jac(a,b):
    inter = min(a[1],b[1]) - max(a[0],b[0]) + 1
    if inter <= 0: return 0.0
    return inter / ((a[1]-a[0]+1) + (b[1]-b[0]+1) - inter)

def match(true, pred):
    cand = sorted(((-jac(t,p), i, j) for i,t in enumerate(true) for j,p in enumerate(pred)
                   if jac(t,p) > 0))
    ut, up, out = set(), set(), {}
    for negv,i,j in cand:
        if i in ut or j in up: continue
        ut.add(i); up.add(j); out[i] = (j, -negv)
    return out

raw = {m: load(m) for m in MODELS}
for m in MODELS:
    assert len(raw[m]) == 20, (m, len(raw[m]))
cells = sorted(set.intersection(*(set(raw[m]) for m in MODELS)))
shared = {}
for c in cells:
    names = None
    for m in MODELS:
        got = {r['name'] for r in raw[m][c]}
        names = got if names is None else names & got
    shared[c] = names
tot = sum(len(v) for v in shared.values())
print(f'[set] {len(cells)} cells, {tot} protein-cells in the common set')
for m in MODELS:
    dropped = sum(len({r['name'] for r in raw[m][c]} - shared[c]) for c in cells)
    allp = sum(len(raw[m][c]) for c in cells)
    print(f'   {m:26s} raw {allp:6d}  dropped {dropped:5d}')

# per-segment records
seg = {m: {} for m in MODELS}
for m in MODELS:
    for c in cells:
        per = {}
        for r in raw[m][c]:
            if r['name'] not in shared[c]: continue
            for task in TASKS:
                true = [tuple(x) for x in r[task]['true']]
                pred = [tuple(x) for x in r[task]['pred']]
                if not true: continue
                mm = match(true, pred) if pred else {}
                for i,(ts,te) in enumerate(true):
                    j, v = mm.get(i, (None, 0.0))
                    per[(r['name'], task, ts, te)] = {
                        'iou': v,
                        'dstart': (pred[j][0]-ts) if j is not None else None,
                        'dend':   (pred[j][1]-te) if j is not None else None,
                        'err': min((max(abs(q[0]-ts), abs(q[1]-te)) for q in pred), default=CAP),
                    }
        seg[m][c] = per

GATES = {'overlap': lambda b,v: b['iou']>0 and v['iou']>0,
         'iou50':   lambda b,v: b['iou']>=0.5 and v['iou']>=0.5,
         'tol3':    lambda b,v: b['err']<=3 and v['err']<=3}

def paired(var, gate, ref=BASE):
    keep = GATES[gate]
    rows = []
    for c in cells:
        b_, v_ = seg[ref][c], seg[var][c]
        di, ds, de, tighter, looser = [], [], [], 0, 0
        for k,b in b_.items():
            v = v_.get(k)
            if v is None or b['dstart'] is None or v['dstart'] is None: continue
            if not keep(b,v): continue
            di.append(v['iou']-b['iou'])
            ds.append(abs(v['dstart'])-abs(b['dstart']))
            de.append(abs(v['dend'])-abs(b['dend']))
            tighter += v['iou']>b['iou']; looser += v['iou']<b['iou']
        rows.append((c[0], len(di), sum(di)/len(di), st.median(di),
                     sum(ds)/len(ds), sum(de)/len(de), tighter/len(di), looser/len(di)))
    byo = defaultdict(list)
    for r in rows: byo[r[0]].append(r[1:])
    po = {}
    for o,rs in byo.items():
        po[o] = [sum(x[i] for x in rs)/len(rs) for i in range(7)]
    ks = sorted(po)
    def col(i): return [po[o][i] for o in ks]
    return {'n': (st.mean(col(0)), st.stdev(col(0))),
            'd_iou': (st.mean(col(1)), st.stdev(col(1)), sum(x>0 for x in col(1)), col(1)),
            'd_med': st.mean(col(2)),
            'd_ds': (st.mean(col(3)), sum(x<0 for x in col(3))),
            'd_de': (st.mean(col(4)), sum(x<0 for x in col(4))),
            'tight': st.mean(col(5)), 'loose': st.mean(col(6))}

print('\n%-24s %-8s %9s %9s %5s %9s %9s %8s %8s %8s' %
      ('model','gate','d_iou','std','impr','n_paired','n_std','d_med','tighter','looser'))
res = {}
for m in MODELS:
    if m == BASE: continue
    for g in ['overlap','iou50','tol3']:
        r = paired(m,g); res[(m,g)] = r
        print('%-24s %-8s %9.6f %9.6f  %d/5 %9.1f %9.1f %8.4f %8.4f %8.4f' %
              (m,g,r['d_iou'][0],r['d_iou'][1],r['d_iou'][2],r['n'][0],r['n'][1],
               r['d_med'],r['tight'],r['loose']))

print('\n--- per-outer d_iou at iou50 ---')
for m in MODELS:
    if m == BASE: continue
    print('%-24s' % m, ['%.6f'%x for x in res[(m,'iou50')]['d_iou'][3]])

print('\n--- displacement at iou50 (negative = tighter) ---')
for m in MODELS:
    if m == BASE: continue
    r = res[(m,'iou50')]
    print('%-24s dstart %+8.4f (%d/5 improved)   dend %+8.4f (%d/5)' %
          (m, r['d_ds'][0], r['d_ds'][1], r['d_de'][0], r['d_de'][1]))

# same-backbone control: esmc6b_full and esmc6b_boundary vs esmc6b_plain
print('\n--- SAME-BACKBONE control: ESM-C variants vs ESM-C plain ---')
for m in ['5cv_esmc6b_boundary','5cv_esmc6b_full']:
    for g in ['iou50','tol3']:
        r = paired(m,g,ref='5cv_esmc6b_plain')
        print('%-24s vs esmc6b_plain %-6s d_iou %+.6f +- %.6f  %d/5  n=%.1f' %
              (m,g,r['d_iou'][0],r['d_iou'][1],r['d_iou'][2],r['n'][0]))
for m in ['5cv_esm2_boundary','5cv_esm2_full']:
    for g in ['iou50','tol3']:
        r = paired(m,g,ref=BASE)
        print('%-24s vs esm2_fp32    %-6s d_iou %+.6f +- %.6f  %d/5  n=%.1f' %
              (m,g,r['d_iou'][0],r['d_iou'][1],r['d_iou'][2],r['n'][0]))
