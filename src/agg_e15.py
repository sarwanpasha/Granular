import json, glob, numpy as np
R = [json.loads(l) for f in sorted(glob.glob('results/e15/l_*.jsonl')) for l in open(f)]
print('rows', len(R), 'datasets', len(set(r['did'] for r in R)))
print('same_partition', sum(r['same_partition'] for r in R), 'of', len(R))
w = [r for r in R if r['lam'] > 1.0001e-3]
ic = np.array([r['iters_cold'] for r in w]); iw = np.array([r['iters_warm'] for r in w])
sc = np.array([r['sec_cold'] for r in w]); sw = np.array([r['sec_warm'] for r in w])
print('total iters cold', int(ic.sum()), 'warm', int(iw.sum()), 'ratio', repr(float(iw.sum() / ic.sum())))
print('total sec cold', repr(float(sc.sum())), 'warm', repr(float(sw.sum())), 'ratio', repr(float(sw.sum() / sc.sum())))
print('median per-step iter ratio', repr(float(np.median(iw / ic))), 'steps faster', int((iw < ic).sum()), 'slower', int((iw > ic).sum()), 'equal', int((iw == ic).sum()))
print('cold steps at cap', int((ic == 20000).sum()), 'warm at cap', int((iw == 20000).sum()))
for d in sorted(set(r['did'] for r in R)):
    rr = sorted([r for r in R if r['did'] == d], key=lambda r: r['lam'])
    best = max(rr, key=lambda r: max(r['acc_rad'], r['acc_den']))
    print(d, 'balls', rr[0]['n_balls'], '->', rr[-1]['n_balls'], 'acc max', round(max(best['acc_rad'], best['acc_den']), 4), 'at lam', round(best['lam'], 4), 'sub95 range', round(min(r['sub95'] for r in rr), 4), round(max(r['sub95'] for r in rr), 4))
