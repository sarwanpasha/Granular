"""Ablations: graph type, label-aware vs unsupervised edges, GB quality (purity, conflict rate).
usage: python e2c_ablate.py <openml_id> <out.jsonl> [nmax] [njobs]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load, uniq, balls, predict
from cgb import CGB, build_graph, incidence, solve_cfuse
from e3_cert import radius_graph
from baselines import gbgpp
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def fit_variant(X, y, lam, var):
    if var.startswith('knn'):
        k = int(var[3:].split('_')[0]); la = not var.endswith('_unsup')
        m = CGB(lam=lam, k=k, label_aware=la).fit(X, y)
        return m.labels_, m.U_
    from sklearn.neighbors import NearestNeighbors
    r = np.median(NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1])
    i, j, w = radius_graph(X, y, r)
    D = incidence(i, j, len(X)); gam = lam * w
    U, Lam, info = solve_cfuse(X, D, gam, 1e-8, 20000)
    Z = Lam + (D @ U) / info['L']; fused = np.linalg.norm(Z, axis=1) <= gam
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(len(X), len(X)))
    return connected_components(A, directed=False)[1], U


def quality(lab, X, y):
    C, R, yb, _ = balls(lab, X, y)
    pur = np.mean([np.mean(y[lab == b] == yb[b]) for b in range(len(C))])
    d = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None], 0))
    inside_other = ((d <= R[None]) & (yb[None, :] != y[:, None])).any(1)
    return float(pur), float(inside_other.mean())


def run(var, X, y, lam, seed=0):
    rng = np.random.default_rng(seed); t0 = time.time()
    if var == 'GBG++':
        g = lambda Xa, ya: (uniq(gbgpp(Xa, ya)), None)
    else:
        g = lambda Xa, ya: fit_variant(Xa, ya, lam, var)
    lab, U = g(X, y); lab = uniq(lab)
    pur, conf = quality(lab, X, y)
    sub = []
    for r in range(3):
        keep = np.sort(rng.choice(len(X), int(0.95 * len(X)), replace=False))
        sub.append(1 - ARI(lab[keep], uniq(g(X[keep], y[keep])[0])))
    jit = [1 - ARI(lab, uniq(g(X + 0.01 * rng.standard_normal(X.shape), y)[0])) for _ in range(3)]
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = uniq(g(X[tr], y[tr])[0]); C, R, yb, dens = balls(m, X[tr], y[tr])
        acc.append(max(np.mean(predict(C, R, yb, dens, X[te], ru) == y[te]) for ru in ('rad', 'den')))
    return dict(variant=var, lam=lam, n_balls=int(lab.max() + 1), purity=pur, conflict=conf,
                sub95=float(np.mean(sub)), jit0_01=float(np.mean(jit)), acc=float(np.mean(acc)), sec=time.time() - t0)


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    X, y, name = load(did, nmax)
    target = int(uniq(gbgpp(X, y)).max() + 1)
    variants = ('knn5', 'knn10', 'knn20', 'radius', 'knn10_unsup')
    def match(v):
        a, b = np.log(1e-3), np.log(10.0)
        for _ in range(12):
            mid = 0.5 * (a + b); nb = int(fit_variant(X, y, float(np.exp(mid)), v)[0].max() + 1)
            if nb > target: a = mid
            else: b = mid
        return float(np.exp(0.5 * (a + b)))
    lams = Parallel(n_jobs=nj)(delayed(match)(v) for v in variants)
    jobs = [('GBG++', None)] + list(zip(variants, lams))
    res = Parallel(n_jobs=nj)(delayed(run)(v, X, y, l) for v, l in jobs)
    with open(outp, 'w') as f:
        for r in res:
            r.update(did=int(did), dataset=name, n=len(X), d=X.shape[1]); f.write(json.dumps(r) + '\n')
    print('done', name)
