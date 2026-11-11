"""E8: streaming granular balls (re-fit on all seen data after each batch; measure churn of old points).
usage: python e8_stream.py <openml_id> <out.jsonl> [nmax] [nbatches] [njobs] [shuffle]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load, uniq, balls, predict, lam_for_target
from cgb import build_graph, incidence, solve_cfuse, CGB
from baselines import gbgpp, gbclassic
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def cgb_fit(X, y, lam, prev=None):
    i, j, w, r = build_graph(X, y)
    D = incidence(i, j, len(X)); gam = lam * w
    Lam0 = None
    if prev is not None:
        pi, pj, pL = prev
        key_prev = {(a, b): k for k, (a, b) in enumerate(zip(pi, pj))}
        Lam0 = np.zeros((len(i), X.shape[1]))
        for k, (a, b) in enumerate(zip(i, j)):
            kk = key_prev.get((a, b))
            if kk is not None:
                v = pL[kk]; nv = np.linalg.norm(v)
                Lam0[k] = v if nv <= gam[k] else v * gam[k] / nv
    U, Lam, info = solve_cfuse(X, D, gam, 1e-8, 20000, Lam0=Lam0)
    nu = 1 / info['L']; Z = Lam + nu * (D @ U)
    fused = np.linalg.norm(Z, axis=1) <= gam
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(len(X), len(X)))
    _, lab = connected_components(A, directed=False)
    return lab, U, (i, j, Lam), info['iters']


def rep(lab, Z):
    nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, Z.shape[1])); np.add.at(C, lab, Z); C /= cnt[:, None]
    return C[lab]


def stream(method, X, y, nb_target, nbatch, lam=None):
    n = len(X); cuts = np.linspace(0, n, nbatch + 1).astype(int)
    prev_lab = prev_rep = None; state = None; rows = []
    for t in range(1, nbatch):
        m = cuts[t]; Xs, ys = X[:m], y[:m]
        t0 = time.time()
        if method == 'CGB':
            lab, U, st, iters = cgb_fit(Xs, ys, lam, state); state = st; Z = rep(lab, U)
        elif method == 'CGB-cold':
            lab, U, _, iters = cgb_fit(Xs, ys, lam, None); Z = rep(lab, U)
        elif method == 'GBG++':
            lab = uniq(gbgpp(Xs, ys)); Z = rep(lab, Xs); iters = 0
        else:
            lab = uniq(gbclassic(Xs, ys, seed=t)); Z = rep(lab, Xs); iters = 0
        sec = time.time() - t0
        C, Rr, yb, dens = balls(lab, Xs, ys)
        nxt = slice(cuts[t], cuts[t + 1])
        acc = max(np.mean(predict(C, Rr, yb, dens, X[nxt], r) == y[nxt]) for r in ('rad', 'den'))
        row = dict(method=method, t=t, n_seen=int(m), n_balls=int(lab.max() + 1), sec=sec, iters=int(iters), acc_next=float(acc))
        if prev_lab is not None:
            k = len(prev_lab)
            row['churn_ari'] = float(1 - ARI(prev_lab, lab[:k]))
            row['drift_rms'] = float(np.sqrt(np.mean(np.sum((Z[:k] - prev_rep) ** 2, 1))))
        prev_lab, prev_rep = lab, Z
        rows.append(row)
    return rows


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    nbatch = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    nj = int(sys.argv[5]) if len(sys.argv) > 5 else os.cpu_count()
    X, y, name = load(did, 10 ** 9)
    if len(sys.argv) > 6 and sys.argv[6] == 'shuffle':
        p = np.random.default_rng(0).permutation(len(X)); X, y = X[p], y[p]
    X, y = X[:nmax], y[:nmax]
    _, y = np.unique(y, return_inverse=True)
    half = len(X) // 2
    nbt = int(uniq(gbgpp(X[:half], y[:half])).max() + 1)
    lam = lam_for_target(X[:half], y[:half], nbt, steps=10)
    res = Parallel(n_jobs=nj)(delayed(stream)(m, X, y, nbt, nbatch, lam) for m in ('CGB', 'CGB-cold', 'GBG++', 'GBclassic'))
    with open(outp, 'w') as f:
        for rr in res:
            for r in rr:
                r.update(did=int(did), dataset=name, lam=lam, order=(sys.argv[6] if len(sys.argv) > 6 else 'natural'))
                f.write(json.dumps(r) + '\n')
    print('done', name)
