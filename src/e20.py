"""E20: Table tab:cifar with the radius graph. Same data (DINOv2 ViT-S/14 CIFAR-10 train, stratified 5000, seed 1),
same perturbations and repetitions as e7_cpu.run_stab, but lambda is searched on [1e-3, 1e5] (18-step log
bisection) so that the radius-graph CGB can reach the GBG++ and GB-classic ball counts.
usage: python e20.py [njobs]
"""
import os, sys, json
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e7_cpu import load, strat, RES
from bench import uniq
from e2c_ablate import fit_variant
from baselines import gbgpp, gbclassic
from sklearn.metrics import adjusted_rand_score as ARI

X, y = load('cifar10_train'); idx = strat(y, 5000, 1); X, y = X[idx], y[idx]; n = len(X)


def stab(f, ref, seed):
    rng = np.random.default_rng(seed); o = {}
    for eps in (0.002, 0.01):
        o[f'jit{eps}'] = float(np.mean([1 - ARI(ref, f(X + eps * rng.standard_normal(X.shape) / np.sqrt(X.shape[1]), y)) for _ in range(3)]))
    v = []
    for _ in range(3):
        keep = np.sort(rng.choice(n, int(0.95 * n), replace=False)); v.append(1 - ARI(ref[keep], f(X[keep], y[keep])))
    o['sub95'] = float(np.mean(v)); return o


def nballs(lam):
    return int(uniq(fit_variant(X, y, lam, 'radius')[0]).max() + 1)


def match(target):
    a, b = np.log(1e-3), np.log(1e5); best = None
    for _ in range(18):
        mid = 0.5 * (a + b); lam = float(np.exp(mid)); nb = nballs(lam)
        if best is None or abs(nb - target) < abs(best[1] - target): best = (lam, nb)
        if nb == target: break
        if nb > target: a = mid
        else: b = mid
    return best


def run(base, fb):
    ref = fb(X, y); nb = int(ref.max() + 1)
    lam, got = match(nb)
    f = lambda A, b: uniq(fit_variant(A, b, lam, 'radius')[0])
    rb = dict(method=base, n_balls=nb, **stab(fb, ref, 0))
    rc = dict(method='CGB-radius', matched_to=base, lam=lam, n_balls=got, **stab(f, f(X, y), 0))
    return [rb, rc]


if __name__ == '__main__':
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = Parallel(n_jobs=nj)(delayed(run)(b, f) for b, f in (('GBG++', lambda A, b: uniq(gbgpp(A, b))),
                                                            ('GBclassic', lambda A, b: uniq(gbclassic(A, b, seed=0)))))
    with open(f'{RES}/e20_cifar10_radius.jsonl', 'w') as fo:
        for rows in out:
            for r in rows:
                fo.write(json.dumps(r) + '\n'); print(json.dumps(r), flush=True)
