"""E21 split into the smallest independent pieces (one solver run per Slurm task), same computation as e21.py.
 cvf  <did> <lam_index> <fold> : inner 3-fold CV (StratifiedKFold(3, shuffle=True, random_state=1)) on the first
                                 25000 training points, ONE fold of ONE lambda -> cv/cvf_<did>_<li>_<fold>.json
 fit  <did> <n> <kind>          : chosen (lambda, rule) from all 39 cvf files; kind = ref (fit on the first n training
                                 points; test accuracy, time) | sub (95% subsample, rng 1) | perm (row permutation, rng 2);
                                 labels saved to res/lab_<did>_<n>_<kind>.npy, row to res/fit_<did>_<n>_<kind>.json
 check <did>                    : GBG++ at n=12500 (data check against the e16w.py rows)
 agg                            : sub95 / order from the saved labels; prints all rows
"""
import sys, os, json, time, glob
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e21
from e21 import data, labels, acc, GRID

D = os.path.dirname(os.path.abspath(__file__))


def choose(did):
    rows = {}
    for f in glob.glob(f'{D}/cv/cvf_{did}_*.json'):
        r = json.load(open(f)); rows.setdefault(r['lam_index'], {})[r['fold']] = r
    assert len(rows) == len(GRID) and all(len(v) == 3 for v in rows.values()), 'incomplete cv'
    cand = []
    for li, v in rows.items():
        for rule in ('rad', 'den'):
            cand.append((round(float(np.mean([v[k]['acc_' + rule] for k in range(3)])), 12), GRID[li], rule))
    return max(cand, key=lambda z: (z[0], z[1]))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'cvf':
        did, li, fold = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); lam = GRID[li]
        X, y, name, Xt, yt, perm = data(did); p = perm[:25000]; X0, y0 = X[p], y[p]
        tr, te = list(StratifiedKFold(3, shuffle=True, random_state=1).split(X0, y0))[fold]
        t0 = time.time(); lab = labels(X0[tr], y0[tr], lam)[0]; s = acc(lab, X0[tr], y0[tr], X0[te], y0[te])
        r = dict(did=int(did), dataset=name, lam=lam, lam_index=li, fold=fold, acc_rad=s['rad'], acc_den=s['den'],
                 n_balls=int(lab.max() + 1), sec=time.time() - t0)
        json.dump(r, open(f'{D}/cv/cvf_{did}_{li:02d}_{fold}.json', 'w')); print(json.dumps(r))
    elif mode == 'fit':
        did, n, kind = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        inner, lam, rule = choose(did)
        X, y, name, Xt, yt, perm = data(did); n = min(n, len(perm)); Xn, yn = X[perm[:n]], y[perm[:n]]
        if kind == 'sub':
            idx = np.sort(np.random.default_rng(1).choice(n, int(0.95 * n), replace=False))
        elif kind == 'perm':
            idx = np.random.default_rng(2).permutation(n)
        else:
            idx = np.arange(n)
        torch.cuda.reset_peak_memory_stats()
        lab, gsec, ssec, it, ne = labels(Xn[idx], yn[idx], lam)
        np.save(f'{D}/res/lab_{did}_{n}_{kind}.npy', lab)
        r = dict(did=int(did), dataset=name, n=n, kind=kind, lam=lam, rule=rule, inner_acc=inner, sec=gsec + ssec,
                 graph_sec=gsec, solve_sec=ssec, iters=it, edges=ne, n_balls=int(lab.max() + 1),
                 gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, gpu=torch.cuda.get_device_name(0))
        if kind == 'ref':
            a = acc(lab, Xn, yn, Xt, yt); r.update(acc=a[rule], acc_rad=a['rad'], acc_den=a['den'])
        json.dump(r, open(f'{D}/res/fit_{did}_{n}_{kind}.json', 'w')); print(json.dumps(r))
    elif mode == 'check':
        from baselines import gbgpp
        did = sys.argv[2]; X, y, name, Xt, yt, perm = data(did)
        Xc, yc = X[perm[:12500]], y[perm[:12500]]; lc = np.unique(gbgpp(Xc, yc), return_inverse=True)[1]
        a = acc(lc, Xc, yc, Xt, yt)
        r = dict(did=int(did), dataset=name, method='GBG++', n=12500, n_balls=int(lc.max() + 1), acc_rad=a['rad'], acc_den=a['den'])
        json.dump(r, open(f'{D}/res/check_{did}.json', 'w')); print(json.dumps(r))
    else:
        for f in sorted(glob.glob(f'{D}/res/fit_*_ref.json')):
            r = json.load(open(f)); did, n = r['did'], r['n']
            ref = np.load(f'{D}/res/lab_{did}_{n}_ref.npy')
            if os.path.exists(f'{D}/res/lab_{did}_{n}_sub.npy'):
                k = np.sort(np.random.default_rng(1).choice(n, int(0.95 * n), replace=False))
                r['sub95'] = float(1 - ARI(ref[k], np.load(f'{D}/res/lab_{did}_{n}_sub.npy')))
            if os.path.exists(f'{D}/res/lab_{did}_{n}_perm.npy'):
                p = np.random.default_rng(2).permutation(n); m3 = np.empty(n, int); m3[p] = np.load(f'{D}/res/lab_{did}_{n}_perm.npy')
                r['order'] = float(1 - ARI(ref, m3))
            print(json.dumps(r))
        for f in sorted(glob.glob(f'{D}/res/check_*.json')):
            print(open(f).read())
