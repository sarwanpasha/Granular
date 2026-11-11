import json, glob, numpy as np
from scipy.stats import wilcoxon
E = {}
for f in glob.glob('results/e18/a_*.jsonl'):
    for l in open(f):
        r = json.loads(l); key = r['method'] if r['method'] != 'CGB-r' else 'CGB|' + r['matched_to']
        E.setdefault(r['did'], {})[key] = r
G = ['GB-acc', 'GB-adp']
for tol in (0.0, 0.05, 0.10):
    rows = []
    for g in G:
        d = sorted(x for x in E if g in E[x] and 'CGB|' + g in E[x] and abs(E[x]['CGB|' + g]['n_balls'] - E[x][g]['n_balls']) <= tol * E[x][g]['n_balls'])
        for k in ('order', 'jit0.01', 'jit0.05', 'sub95', 'acc'):
            a = np.array([E[x][g][k] for x in d]); c = np.array([E[x]['CGB|' + g][k] for x in d])
            w = int((c > a).sum() if k == 'acc' else (c < a).sum()); t = int((c == a).sum())
            p = float(wilcoxon(c, a).pvalue) if np.any(c != a) and len(d) > 5 else 1.0
            rows.append([g, len(d), k, float(np.median(a)), float(np.median(c)), w, t, len(c) - w - t, p])
    m = len(rows); o = np.argsort([r[-1] for r in rows]); h = [0] * m; run = 0
    for q, i in enumerate(o):
        run = max(run, min(1, (m - q) * rows[i][-1])); h[i] = run
    print('=== ball-count tolerance', tol)
    for r, x in zip(rows, h):
        print(r[0], 'n=%d' % r[1], r[2], 'gen %.4f CGB %.4f' % (r[3], r[4]), '%d/%d/%d' % tuple(r[5:8]), 'holm %.2g' % x)
