"""E15: lambda path for the radius-graph CGB (sensitivity + warm-start cost).
For 16 log-spaced lambda values in [1e-3, 300], in increasing order:
  * cold solve (dual initialised at 0) and warm solve (dual initialised at the previous lambda's dual scaled by
    lam_new/lam_old, which keeps it feasible); iterations, seconds, and whether both give the same partition;
  * number of balls, 5-fold CV accuracy (radius and density rules), sub95 and jit0.01 instability (3 repetitions).
usage: python e15.py <openml_id> <out.jsonl> <nmax>
"""
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load, balls, predict
from cgb import incidence, solve_cfuse
from e3_cert import radius_graph


def graph(X, y):
    r = np.median(NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1])
    return radius_graph(X, y, r)


def labels(X, i, j, w, lam, Lam0=None):
    n = len(X); D = incidence(i, j, n); gam = lam * w
    t0 = time.time(); U, Lam, info = solve_cfuse(X, D, gam, 1e-8, 20000, Lam0=Lam0); sec = time.time() - t0
    Z = Lam + (D @ U) / info['L']; fused = np.linalg.norm(Z, axis=1) <= gam
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(n, n))
    return connected_components(A, directed=False)[1], Lam, info['iters'], sec


def fitlab(X, y, lam):
    i, j, w = graph(X, y); return labels(X, i, j, w, lam)[0]


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    X, y, name = load(did, nmax); n = len(X)
    i, j, w = graph(X, y)
    lams = np.exp(np.linspace(np.log(1e-3), np.log(300.0), 16))
    prev = None; out = []; fo = open(outp, 'w')
    for lam in lams:
        lab_c, Lam_c, it_c, sec_c = labels(X, i, j, w, lam)
        if prev is None:
            lab_w, it_w, sec_w = lab_c, it_c, sec_c
        else:
            lab_w, _, it_w, sec_w = labels(X, i, j, w, lam, Lam0=prev[1] * (lam / prev[0]))
        prev = (lam, Lam_c)
        r = dict(did=int(did), dataset=name, n=n, d=X.shape[1], lam=float(lam), n_balls=int(lab_c.max() + 1),
                 iters_cold=int(it_c), iters_warm=int(it_w), sec_cold=sec_c, sec_warm=sec_w,
                 same_partition=bool(ARI(lab_c, lab_w) == 1.0))
        rng = np.random.default_rng(15)
        r['sub95'] = float(np.mean([1 - ARI(lab_c[k], fitlab(X[k], y[k], lam)) for k in
                                    [np.sort(rng.choice(n, int(0.95 * n), replace=False)) for _ in range(3)]]))
        r['jit0.01'] = float(np.mean([1 - ARI(lab_c, fitlab(X + 0.01 * rng.standard_normal(X.shape), y, lam)) for _ in range(3)]))
        acc = {'rad': [], 'den': []}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            C, Rr, yb, dens = balls(fitlab(X[tr], y[tr], lam), X[tr], y[tr])
            for rule in acc: acc[rule].append(np.mean(predict(C, Rr, yb, dens, X[te], rule) == y[te]))
        r['acc_rad'] = float(np.mean(acc['rad'])); r['acc_den'] = float(np.mean(acc['den']))
        fo.write(json.dumps(r) + '\n'); fo.flush(); print(json.dumps(r), flush=True)
