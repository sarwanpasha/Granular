import json, glob, sys, numpy as np
from scipy.stats import wilcoxon
R = [json.loads(l) for f in glob.glob(sys.argv[1] + '/a_*.jsonl') for l in open(f)]
skip = set(int(x) for x in sys.argv[2:])
dids = sorted(set(r['did'] for r in R) - skip)
print('datasets', len(dids))
def V(m, b, k):
    return np.array([np.mean([r[k] for r in R if r['did'] == d and r['method'] == m and r['matched_to'] == b]) for d in dids])
rows = []
for k in ('rad', 'den', 'k3'):
    for b in ('GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s'):
        base, cgb = V(b, b, k), V('CGB-r', b, k)
        w = int((cgb > base).sum()); t = int((cgb == base).sum())
        p = float(wilcoxon(cgb, base).pvalue) if np.any(cgb != base) else 1.0
        rows.append([k, b, float(np.median(base)), float(np.median(cgb)), w, t, len(dids) - w - t, p])
m = len(rows); order = np.argsort([r[-1] for r in rows]); holm = [0.0] * m; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * rows[i][-1])); holm[i] = run
for r, h in zip(rows, holm):
    print(r[0], r[1], 'base', round(r[2], 4), 'cgb', round(r[3], 4), f'{r[4]}/{r[5]}/{r[6]}', 'p', f'{r[7]:.2g}', 'holm', f'{h:.2g}')
