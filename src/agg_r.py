import json, glob
import numpy as np
from scipy.stats import wilcoxon
B = {}; Rr = {}
for f in glob.glob('results/bench/b_*.jsonl'):
    for l in open(f):
        r = json.loads(l); B.setdefault(r['did'], []).append(r)
for f in glob.glob('results/bench_r/r_*.jsonl'):
    for l in open(f):
        r = json.loads(l); Rr.setdefault(r['did'], []).append(r)
keys = ['order', 'jit0.01', 'sub95', 'drift0.01', 'acc']
print('datasets with radius rerun:', len(Rr))
for base in ['GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s']:
    bb, rr, kk = [], [], []
    for did in Rr:
        b = [x for x in B.get(did, []) if x['method'] == base]
        r = [x for x in Rr[did] if x['matched_to'] == base]
        k = [x for x in B.get(did, []) if x['method'] == 'CGB' and x.get('matched_to') == base]
        if b and r and k: bb.append(b[0]); rr.append(r[0]); kk.append(k[0])
    print(f'== vs {base} n={len(bb)}  balls base {np.median([x["n_balls"] for x in bb]):.0f} cgbr {np.median([x["n_balls"] for x in rr]):.0f}')
    for key in keys:
        bv = np.array([x[key] for x in bb]); rv = np.array([x[key] for x in rr]); kv = np.array([x[key] for x in kk])
        win = np.sum(rv > bv) if key == 'acc' else np.sum(rv < bv)
        try: p = wilcoxon(rv, bv).pvalue
        except Exception: p = float('nan')
        print(f'  {key:9s} base {np.median(bv):.3f} cgb-knn {np.median(kv):.3f} cgb-r {np.median(rv):.3f} wins {win}/{len(bb)} p={p:.1e}')
