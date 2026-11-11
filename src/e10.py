"""E10: does a better ball-based predictor change the accuracy comparison?
Same data, folds (StratifiedKFold(5, shuffle, random_state=0)) and matched lambdas as the radius-graph
benchmark (results/bench_r). For every baseline B and fold, fit B and CGB-r (lambda matched to B, read
from results/bench_r/r_<id>.jsonl) on the training fold and evaluate three predictors, each fixed a
priori and applied identically to every generator:
  rad : label of the ball minimising ||x-c|| - rho            (existing)
  den : label of the ball minimising ||x-c|| - n_b/n           (existing)
  k3  : size-weighted vote of the 3 balls with smallest ||x-c|| - rho   (new)
No predictor is selected on test data; all three are reported.
usage: python e10.py <openml_id> <out.jsonl> <nmax> [njobs]
"""
import sys, os, json
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from bench import balls, predict
import bench_r

BASES = ('GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s')


def k3(C, Rr, yb, cnt, Xt, K):
    out = np.empty(len(Xt), int)
    for s in range(0, len(Xt), 1024):
        Z = Xt[s:s + 1024]
        d = np.sqrt(np.maximum((Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None], 0)) - Rr[None]
        k = min(3, len(C)); idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        V = np.zeros((len(Z), K))
        for j in range(k):
            np.add.at(V, (np.arange(len(Z)), yb[idx[:, j]]), cnt[idx[:, j]])
        out[s:s + 1024] = V.argmax(1)
    return out


def evalb(lab, Xtr, ytr, Xte, yte, K):
    C, Rr, yb, dens = balls(lab, Xtr, ytr)
    cnt = np.bincount(bench.uniq(lab))
    return dict(rad=float(np.mean(predict(C, Rr, yb, dens, Xte, 'rad') == yte)),
                den=float(np.mean(predict(C, Rr, yb, dens, Xte, 'den') == yte)),
                k3=float(np.mean(k3(C, Rr, yb, cnt, Xte, K) == yte)), n_balls=int(len(C)))


def fold(X, y, tr, te, lams, f):
    K = int(y.max() + 1); out = []
    for b in BASES:
        lab = bench.gen(b, X[tr], y[tr], 0, None)[0]
        r = dict(fold=f, method=b, matched_to=b); r.update(evalb(lab, X[tr], y[tr], X[te], y[te], K)); out.append(r)
        if b in lams:
            lab = bench_r.gen_r('CGB-r', X[tr], y[tr], 0, lams[b])[0]
            r = dict(fold=f, method='CGB-r', matched_to=b, lam=lams[b]); r.update(evalb(lab, X[tr], y[tr], X[te], y[te], K)); out.append(r)
    return out


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    X, y, name = bench.load(did, nmax)
    lams = {r['matched_to']: r['lam'] for r in map(json.loads, open(f'results/bench_r/r_{did}.jsonl'))}
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    res = Parallel(n_jobs=nj)(delayed(fold)(X, y, tr, te, lams, f) for f, (tr, te) in enumerate(skf.split(X, y)))
    with open(outp, 'w') as fo:
        for rr in res:
            for r in rr:
                r.update(did=int(did), dataset=name, n=len(X), d=X.shape[1]); fo.write(json.dumps(r) + '\n')
    print('done', name)
