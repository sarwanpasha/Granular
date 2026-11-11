import json, glob, numpy as np
from scipy.stats import wilcoxon
E = {}
for f in glob.glob('results/e17/a_*.jsonl'):
    for l in open(f):
        r = json.loads(l); E.setdefault(r['did'], {})[r['method']] = r
R = {}
for f in glob.glob('results/bench_r/r_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r['matched_to'] == 'GBG++': R[r['did']] = r
M = ['KM-class', 'Ward-class', 'KCenter-class']
dids = sorted(d for d in E if d in R and all(m in E[d] for m in M))
rows = []
for m in M:
    for k in ('order', 'jit0.01', 'jit0.05', 'sub95', 'acc'):
        a = np.array([E[d][m][k] for d in dids]); c = np.array([R[d][k] for d in dids])
        w = int((c > a).sum() if k == 'acc' else (c < a).sum()); t = int((c == a).sum())
        p = float(wilcoxon(c, a).pvalue) if np.any(c != a) else 1.0
        rows.append([m, k, float(np.median(a)), float(np.median(c)), w, t, len(c) - w - t, p])
mm = len(rows); o = np.argsort([r[-1] for r in rows]); h = [0] * mm; run = 0
for q, i in enumerate(o):
    run = max(run, min(1, (mm - q) * rows[i][-1])); h[i] = run
print('datasets', len(dids), 'tests', mm)
for r, x in zip(rows, h):
    print(r[0], r[1], '%.4f %.4f' % (r[2], r[3]), '%d/%d/%d' % tuple(r[4:7]), 'holm %.2g' % x)
nb = [E[d]['KM-class']['n_balls'] - R[d]['n_balls'] for d in dids]
print('ballcount diff (KM - CGB) min/median/max', min(nb), float(np.median(nb)), max(nb))
