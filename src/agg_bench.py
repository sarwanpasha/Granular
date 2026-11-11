import json, glob, sys
import numpy as np
from scipy.stats import wilcoxon
rows = []
for f in sorted(glob.glob('results/bench/b_*.jsonl')):
    rows += [json.loads(l) for l in open(f)]
by = {}
for r in rows:
    by.setdefault(r['did'], []).append(r)
keys = ['order', 'jit0.01', 'jit0.05', 'sub95', 'drift0.01', 'drift0.05', 'acc']
print('datasets complete:', len(by))
for base in ['GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s']:
    B, C = [], []
    for did, rs in by.items():
        b = [r for r in rs if r['method'] == base]; c = [r for r in rs if r['method'] == 'CGB' and r.get('matched_to') == base]
        if b and c: B.append(b[0]); C.append(c[0])
    if not B: continue
    print(f'== CGB vs {base}  (n={len(B)} datasets; median balls base {np.median([b["n_balls"] for b in B]):.0f} cgb {np.median([c["n_balls"] for c in C]):.0f})')
    for k in keys:
        bv = np.array([b[k] for b in B]); cv = np.array([c[k] for c in C])
        win = np.sum(cv > bv) if k == 'acc' else np.sum(cv < bv)
        tie = np.sum(np.isclose(cv, bv))
        try: p = wilcoxon(cv, bv).pvalue
        except Exception: p = float('nan')
        print(f'  {k:10s} base {np.median(bv):.3f}  cgb {np.median(cv):.3f}  wins {win}/{len(B)} ties {tie}  p={p:.1e}')
refs = [r for r in rows if r['method'] == 'REF']
if refs:
    print('REF medians: ' + ' '.join(f'{k} {np.median([r[k] for r in refs]):.3f}' for k in ('kNN5', 'SVM', 'RF')))
