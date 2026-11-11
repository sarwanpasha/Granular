import json, glob, numpy as np
from scipy.stats import wilcoxon
R = [json.loads(l) for f in glob.glob('results/e12b/a_*.jsonl') for l in open(f)]
dids = sorted(set(r['did'] for r in R)); G = ['GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s']
print('datasets', len(dids))
def V(g, k): return np.array([np.mean([r[k] for r in R if r['did'] == d and r['gen'] == g]) for d in dids])
rows = []
for l in ('GB-NB', 'GB-kNN', 'GB-SVM', 'GB-LR'):
    for v in ('order', 'seed', 'sub95', 'acc'):
        k = f'acc_{l}' if v == 'acc' else f'chg_{v}_{l}'
        c = V('CGB-r', k)
        for g in G:
            b = V(g, k); hi = v == 'acc'
            w = int((c > b).sum() if hi else (c < b).sum()); t = int((c == b).sum())
            p = float(wilcoxon(c, b).pvalue) if np.any(c != b) else 1.0
            rows.append([l, v, g, float(np.median(b)), float(np.median(c)), w, t, len(c) - w - t, p])
m = len(rows); order = np.argsort([r[-1] for r in rows]); holm = [0.0] * m; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * rows[i][-1])); holm[i] = run
for r, h in zip(rows, holm):
    print(r[0], r[1], r[2], 'base', round(r[3], 4), 'cgb', round(r[4], 4), f'{r[5]}/{r[6]}/{r[7]}', 'p', f'{r[8]:.2g}', 'holm', f'{h:.2g}')
json.dump([r + [h] for r, h in zip(rows, holm)], open('results/agg_e12b.json', 'w'))
