import json, glob, numpy as np
R = [json.loads(l) for f in sorted(glob.glob('results/e14/c_*.jsonl')) for l in open(f)]
print('rows', len(R), 'datasets', len(set(r['did'] for r in R)))
ok = [r for r in R if r['emp_break_min']]
for k in ('rho_paper', 'rho_tree', 'rho_er'):
    pos = [r for r in ok if r[k] > 0]
    rat = np.array([r['emp_break_min'] / r[k] for r in pos])
    print(k, 'positive', len(pos), 'of', len(ok), 'median ratio emp/cert', repr(float(np.median(rat))) if len(rat) else None, 'min', repr(float(rat.min())) if len(rat) else None)
print('er/tree median', repr(float(np.median([r['rho_er'] / r['rho_tree'] for r in ok if r['rho_tree'] > 0]))))
print('cert_er_holds', sum(r.get('cert_er_holds') is True for r in R), 'of', sum('cert_er_holds' in r for r in R))
for d in (6, 5, 4):
    print('round', d, 'certified', sum(r[f'round{d}_cert_er'] for r in R), 'same', sum(r[f'round{d}_same'] for r in R), 'of', len(R), 'median norm', repr(float(np.median([r[f'round{d}_norm'] for r in R]))))
