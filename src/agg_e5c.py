import json, glob, numpy as np, sys
from scipy.stats import wilcoxon
src, keys = sys.argv[1], sys.argv[2:]
R = [json.loads(l) for f in glob.glob(src) for l in open(f)]
print('rows', len(R), 'datasets', len(set(r['did'] for r in R)))
for eps in sorted(set(r['eps'] for r in R)):
    D = {}
    for r in R:
        if r['eps'] == eps: D.setdefault(r['did'], []).append(r)
    dids = sorted(D)
    for met in ('acc', 'disagree'):
        M = {m: np.array([np.mean([x[m + '_' + met] for x in D[d] if m + '_' + met in x]) for d in dids]) for m in keys}
        line = f'eps {eps} {met}: ' + ' '.join(f'{m} {np.median(M[m]):.4f}' for m in keys)
        c = M['CGB-sig']
        for m in keys:
            if m == 'CGB-sig': continue
            better = c > M[m] if met == 'acc' else c < M[m]
            w = int(better.sum()); t = int((c == M[m]).sum()); p = wilcoxon(c, M[m]).pvalue if np.any(c != M[m]) else 1.0
            line += f' | vs {m} {w}/{t}/{len(c)-w-t} p {p:.2g}'
        print(line)
