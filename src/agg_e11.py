import json, glob, numpy as np
from scipy.stats import wilcoxon
E = {}
for f in glob.glob('results/e11/a_*.jsonl'):
    for l in open(f):
        r = json.loads(l); E.setdefault(r['did'], {})[r['method']] = r
R = {}
for f in glob.glob('results/bench_r/r_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r['matched_to'] == 'GBG++': R[r['did']] = r
B = {}
for f in glob.glob('results/bench/b_*.jsonl'):
    for l in open(f):
        r = json.loads(l); B.setdefault(r['did'], {})[r['method']] = r
dids = sorted(d for d in E if d in R)
print('datasets', len(dids))
for m in ['GBG++-canon', 'GBclassic-canon', 'RGGB-s-canon', 'ScOrGBG-s-canon', 'SON-class', 'CGB-r-unif', 'CGB-knn-gauss']:
    line = m
    for k in ('order', 'jit0.01', 'jit0.05', 'sub95', 'acc'):
        a = np.array([E[d][m][k] for d in dids]); c = np.array([R[d][k] for d in dids])
        w = int((c > a).sum() if k == 'acc' else (c < a).sum()); t = int((c == a).sum())
        p = wilcoxon(c, a).pvalue if np.any(c != a) else 1.0
        line += f' | {k} {np.median(a):.4f} vs CGB-r {np.median(c):.4f} {w}/{t}/{len(c)-w-t} p {p:.2g}'
    print(line)
