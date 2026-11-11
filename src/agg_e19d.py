import json, glob, numpy as np
from scipy.stats import wilcoxon
E = {}
for f in glob.glob('results/e19d/a_*.jsonl'):
    r = json.loads(open(f).readline()); E[r['did']] = r
B = {}
for f in glob.glob('results/bench/b_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r['method'] == 'GBG++': B[r['did']] = r
d = sorted(x for x in E if x in B)
print('datasets', len(d))
rows = []
def test(name, c, g, hi):
    c, g = np.array(c), np.array(g)
    w = int((c > g).sum() if hi else (c < g).sum()); t = int((c == g).sum())
    p = float(wilcoxon(c, g).pvalue) if np.any(c != g) else 1.0
    rows.append([name, float(np.median(g)), float(np.median(c)), w, t, len(c) - w - t, p])
test('acc_nested', [E[x]['acc_cv'] for x in d], [E[x]['g_acc_cv'] for x in d], True)
for k in ('order', 'jit0.01', 'jit0.05', 'sub95'):
    test(k, [E[x]['stab'][k] for x in d], [B[x][k] for x in d], False)
m = len(rows); o = np.argsort([r[-1] for r in rows]); h = [0] * m; run = 0
for q, i in enumerate(o):
    run = max(run, min(1, (m - q) * rows[i][-1])); h[i] = run
for r, x in zip(rows, h):
    print(r[0], 'GBG++ %.4f CGB %.4f' % (r[1], r[2]), '%d/%d/%d' % tuple(r[3:6]), 'p %.2g holm %.2g' % (r[6], x))
lam = [E[x]['lam_star'] for x in d]
print('lam_star counts', {l: lam.count(l) for l in sorted(set(lam))})
print('rule_star', {r: [E[x]['rule_star'] for x in d].count(r) for r in ('rad', 'den')})
nbc = np.array([E[x]['stab']['n_balls'] for x in d]); nbg = np.array([B[x]['n_balls'] for x in d])
print('balls median CGB %.1f GBG++ %.1f ; ratio median %.4f' % (np.median(nbc), np.median(nbg), np.median(nbc / nbg)))
fl = [len(set(f['lam'] for f in E[x]['folds'])) for x in d]
print('distinct lam across 5 outer folds', {k: fl.count(k) for k in sorted(set(fl))})
