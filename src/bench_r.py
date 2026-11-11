"""Re-run the E1/E2 benchmark CGB arm with the RADIUS graph (theory setting; best variant in E2c).
Matches each baseline's ball count read from results/bench/b_<id>.jsonl. usage: bench_r.py <id> <out> [nmax] [njobs]"""
import sys, os, json
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from e2c_ablate import fit_variant
from bench import uniq


def gen_r(method, X, y, seed=0, lam=None):
    lab, U = fit_variant(X, y, lam, 'radius'); lab = uniq(lab)
    nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, U); C /= np.maximum(cnt, 1)[:, None]
    return lab, C[lab]


_ORIG = bench.gen


def gen_any(method, X, y, seed=0, lam=None):
    return gen_r(method, X, y, seed, lam) if method == 'CGB-r' else _ORIG(method, X, y, seed, lam)


def ev(X, y, lam, seed):
    bench.gen = gen_any  # patch inside the worker process
    return bench.evaluate('CGB-r', X, y, lam, seed)


def lam_r(X, y, target):
    a, b = np.log(1e-3), np.log(300.0); best = None
    for _ in range(14):
        mid = 0.5 * (a + b); lam = float(np.exp(mid)); nb = int(fit_variant(X, y, lam, 'radius')[0].max() + 1)
        if best is None or abs(nb - target) < abs(best[1] - target): best = (lam, nb)
        if nb == target: break
        if nb > target: a = mid
        else: b = mid
    return best[0]


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    X, y, name = bench.load(did, nmax)
    rows = [json.loads(l) for l in open(f'results/bench/b_{did}.jsonl')]
    targets = {r['method']: r['n_balls'] for r in rows if r['method'] in ('GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s')}
    lams = Parallel(n_jobs=nj)(delayed(lam_r)(X, y, t) for t in targets.values())
    res = Parallel(n_jobs=nj)(delayed(ev)(X, y, l, 7 + i) for i, l in enumerate(lams))
    with open(outp, 'w') as f:
        for (b, t), l, r in zip(targets.items(), lams, res):
            r.update(did=int(did), dataset=name, n=len(X), d=X.shape[1], method='CGB-r', lam=l, matched_to=b)
            f.write(json.dumps(r) + '\n')
    print('done', name)
