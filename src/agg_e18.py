import json, glob, numpy as np
from scipy.stats import wilcoxon
E = {}
for f in glob.glob('results/e18/a_*.jsonl'):
    for l in open(f):
        r = json.loads(l); key = r['method'] if r['method'] != 'CGB-r' else 'CGB|' + r['matched_to']
        E.setdefault(r['did'], {})[key] = r
G = ['GB-acc', 'GB-adp']
d = sorted(x for x in E if all(g in E[x] and 'CGB|' + g in E[x] for g in G))
print('datasets', len(d))
rows = []
for g in G:
    for k in ('order', 'seed', 'jit0.01', 'jit0.05', 'sub95', 'acc'):
        if not all(k in E[x][g] and k in E[x]['CGB|' + g] for x in d): continue
        a = np.array([E[x][g][k] for x in d]); c = np.array([E[x]['CGB|' + g][k] for x in d])
        w = int((c > a).sum() if k == 'acc' else (c < a).sum()); t = int((c == a).sum())
        p = float(wilcoxon(c, a).pvalue) if np.any(c != a) else 1.0
        rows.append([g, k, float(np.median(a)), float(np.median(c)), w, t, len(c) - w - t, p])
m = len(rows); o = np.argsort([r[-1] for r in rows]); h = [0] * m; run = 0
for q, i in enumerate(o):
    run = max(run, min(1, (m - q) * rows[i][-1])); h[i] = run
print('tests', m)
for r, x in zip(rows, h):
    print(r[0], r[1], 'gen %.4f CGB %.4f' % (r[2], r[3]), '%d/%d/%d' % tuple(r[4:7]), 'holm %.2g' % x)
for g in G:
    nb = np.array([E[x][g]['n_balls'] for x in d]); nc = np.array([E[x]['CGB|' + g]['n_balls'] for x in d])
    print(g, 'balls median gen %.1f CGB %.1f; exact match %d; median |rel diff| %.4f; gen_sec median gen %.4f CGB %.4f' % (np.median(nb), np.median(nc), int((nb == nc).sum()), np.median(np.abs(nc - nb) / nb), np.median([E[x][g]['gen_sec'] for x in d]), np.median([E[x]['CGB|' + g]['gen_sec'] for x in d])))
