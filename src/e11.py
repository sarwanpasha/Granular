"""E11: reviewer-objection controls, same protocol as the benchmark (bench.evaluate: order, jitter 0.01/0.05,
sub95, drift, 5-fold CV accuracy).
 (1) canonicalised greedy generators: rows sorted lexicographically (features, then label) and seed fixed to 0
     before every call, so row order and seeds cannot matter; what remains is genuine data sensitivity.
     Methods: GBG++-canon, GBclassic-canon, RGGB-s-canon, ScOrGBG-s-canon (no lambda; native granularity).
 (2) plain sum-of-norms clustering run separately on each class (Hocking et al. / Chi & Lange): within-class
     10-NN graph, Gaussian weights exp(-d^2 / median(d^2)), no shared graph: SON-class.
 (3) ablations of CGB's graph and kernel: CGB-r-unif (radius graph, uniform weights),
     CGB-knn-gauss (all-points 10-NN graph, Gaussian weights, cross-label edges removed).
 Lambda-based methods are matched to the GBG++ ball count of the same data set (14-step bisection on
 log lambda in [1e-3, 300], as in the benchmark).
usage: python e11.py <openml_id> <out.jsonl> <nmax> [njobs]
"""
import sys, os, json
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from bench import uniq
from cgb import incidence, solve_cfuse
from baselines import gbgpp, gbclassic, rggb_s, scorgbg_s
from e3_cert import radius_graph

GREEDY = {'GBG++': lambda X, y: gbgpp(X, y), 'GBclassic': lambda X, y: gbclassic(X, y, seed=0),
          'RGGB-s': lambda X, y: rggb_s(X, y, seed=0), 'ScOrGBG-s': lambda X, y: scorgbg_s(X, y, seed=0)}


def canon(base, X, y):
    o = np.lexsort(np.c_[X, y].T[::-1])          # sort by first feature, then second, ..., then label
    lab = np.empty(len(X), int); lab[o] = uniq(GREEDY[base](X[o], y[o])); return lab


def fuse(X, i, j, w, lam):
    n = len(X)
    if len(i) == 0:
        return np.arange(n), X.copy()
    D = incidence(i, j, n); gam = lam * w
    U, Lam, info = solve_cfuse(X, D, gam, 1e-8, 20000)
    Z = Lam + (D @ U) / info['L']; fused = np.linalg.norm(Z, axis=1) <= gam
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(n, n))
    return connected_components(A, directed=False)[1], U


def knn_edges(X, k=10):
    k = min(k, len(X) - 1)
    if k < 1:
        return np.array([], int), np.array([], int), np.array([])
    d, ind = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    r = np.repeat(np.arange(len(X)), k); c = ind[:, 1:].ravel(); dd = d[:, 1:].ravel()
    a = np.minimum(r, c); b = np.maximum(r, c); key, first = np.unique(a.astype(np.int64) * len(X) + b, return_index=True)
    return a[first], b[first], dd[first]


def son_class(X, y, lam):
    lab = np.empty(len(X), int); U = np.empty_like(X); off = 0
    for c in np.unique(y):
        idx = np.where(y == c)[0]; Xc = X[idx]
        i, j, d = knn_edges(Xc)
        w = np.exp(-d ** 2 / (np.median(d ** 2) + 1e-12)) if len(d) else d
        l, Uc = fuse(Xc, i, j, w, lam); lab[idx] = l + off; off += l.max() + 1; U[idx] = Uc
    return lab, U


def cgb_unif(X, y, lam):
    r = np.median(NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1])
    i, j, w = radius_graph(X, y, r)
    return fuse(X, i, j, np.ones_like(w), lam)


def cgb_knn_gauss(X, y, lam):
    i, j, d = knn_edges(X); keep = y[i] == y[j]; i, j, d = i[keep], j[keep], d[keep]
    w = np.exp(-d ** 2 / (np.median(d ** 2) + 1e-12))
    return fuse(X, i, j, w, lam)


LAMF = {'SON-class': son_class, 'CGB-r-unif': cgb_unif, 'CGB-knn-gauss': cgb_knn_gauss}
_ORIG = bench.gen


def gen_e11(method, X, y, seed=0, lam=None):
    if method.endswith('-canon'):
        lab = canon(method[:-6], X, y); Z = X
    elif method in LAMF:
        lab, Z = LAMF[method](X, y, lam); lab = uniq(lab)
    else:
        return _ORIG(method, X, y, seed, lam)
    nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, Z); C /= np.maximum(cnt, 1)[:, None]
    return lab, C[lab]


def lam_match(f, X, y, target):
    a, b = np.log(1e-3), np.log(300.0); best = None
    for _ in range(14):
        mid = 0.5 * (a + b); lam = float(np.exp(mid)); nb = int(uniq(f(X, y, lam)[0]).max() + 1)
        if best is None or abs(nb - target) < abs(best[1] - target): best = (lam, nb)
        if nb == target: break
        if nb > target: a = mid
        else: b = mid
    return best[0]


def job(method, X, y, did, target):
    bench.gen = gen_e11
    lam = lam_match(LAMF[method], X, y, target) if method in LAMF else None
    r = bench.evaluate(method, X, y, lam, 11)
    r.update(method=method, lam=lam, target=target, did=int(did))
    return r


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    X, y, name = bench.load(did, nmax)
    target = [json.loads(l) for l in open(f'results/bench/b_{did}.jsonl') if json.loads(l)['method'] == 'GBG++'][0]['n_balls']
    meths = ['GBG++-canon', 'GBclassic-canon', 'RGGB-s-canon', 'ScOrGBG-s-canon', 'SON-class', 'CGB-r-unif', 'CGB-knn-gauss']
    res = Parallel(n_jobs=nj)(delayed(job)(m, X, y, did, target) for m in meths)
    with open(outp, 'w') as f:
        for r in res:
            r.update(dataset=name, n=len(X), d=X.shape[1]); f.write(json.dumps(r) + '\n')
    print('done', name)
