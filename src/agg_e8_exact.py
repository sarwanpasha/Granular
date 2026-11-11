import json, glob, numpy as np
from scipy.stats import wilcoxon
R = [json.loads(l) for f in sorted(glob.glob('results/e8/s_*.jsonl')) for l in open(f)]
keys = sorted(set((r['did'], r['order']) for r in R)); meths = sorted(set(r['method'] for r in R))
print('streams', len(keys), 'methods', meths, 'orders', sorted(set(r['order'] for r in R)))
def V(m, k):
    return np.array([np.mean([r[k] for r in R if (r['did'], r['order']) == s and r['method'] == m and k in r]) for s in keys])
for k, low in (('churn_ari', True), ('drift_rms', True), ('acc_next', False), ('sec', True)):
    c = V('CGB', k); line = k + ' ' + ' '.join(f'{m} {np.median(V(m, k)):.4f}' for m in meths)
    for m in meths:
        if m == 'CGB': continue
        b = V(m, k); w = int(((c < b) if low else (c > b)).sum()); t = int((c == b).sum())
        p = wilcoxon(c, b).pvalue if np.any(c != b) else 1.0
        line += f' | {m} {w}/{t}/{len(c)-w-t} p {p:.2g}'
    print(line)
