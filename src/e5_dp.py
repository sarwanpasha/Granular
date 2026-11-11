"""E5: differentially private granular balls via noisy anchor statistics + post-processing.
usage: python e5_dp.py <openml_id> <out.jsonl> [reps] [njobs]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load, uniq, balls, predict, lam_for_target
from cgb import CGB
from baselines import gbgpp, gbclassic

EPSS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)
DELTA = 1e-5


def gauss_sigma(eps, delta, sens):
    """Analytic Gaussian mechanism (Balle & Wang 2018)."""
    if np.isinf(eps):
        return 0.0
    from scipy.stats import norm
    def dl(s):
        return norm.cdf(sens / (2 * s) - eps * s / sens) - np.exp(eps) * norm.cdf(-sens / (2 * s) - eps * s / sens)
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if dl(mid) > delta: lo = mid
        else: hi = mid
    return hi


def gb_acc(mem, Xp, yp, Xte, yte):
    C, Rr, yb, dens = balls(mem, Xp, yp)
    return max(float(np.mean(predict(C, Rr, yb, dens, Xte, r) == yte)) for r in ('rad', 'den'))


def one(X, y, rep, M):
    rng = np.random.default_rng(rep); n, d = X.shape; K = int(y.max() + 1)
    p = rng.permutation(n); npub = n // 10
    pub, prv, te = p[:npub], p[npub:npub + int(0.65 * n)], p[npub + int(0.65 * n):]
    A = KMeans(min(M, npub), n_init=1, random_state=rep).fit(X[pub]).cluster_centers_
    a = ((X[prv] ** 2).sum(1)[:, None] - 2 * X[prv] @ A.T + (A ** 2).sum(1)[None]).argmin(1)
    cell = a * K + y[prv]; G = len(A) * K
    cnt = np.bincount(cell, minlength=G).astype(float)
    S = np.zeros((G, d)); np.add.at(S, cell, X[prv] / np.sqrt(d))
    out = []
    for eps in EPSS:
        sig = gauss_sigma(eps, DELTA, np.sqrt(2.0))
        cn = cnt + sig * rng.standard_normal(G)
        Sn = S + sig * rng.standard_normal(S.shape)
        tau = max(3.0, 3 * sig)
        keep = cn > tau
        if keep.sum() < 2 * K:
            keep = np.argsort(-cn)[:2 * K]; keep = np.isin(np.arange(G), keep)
        Xp = np.clip(Sn[keep] * np.sqrt(d) / cn[keep, None], 0, 1); yp = (np.arange(G) % K)[keep]
        res = dict(M=M, rep=rep, eps=float(eps) if np.isfinite(eps) else 'inf', n_cells=int(keep.sum()), sigma=sig)
        cs = np.zeros((K, d)); cc = np.zeros(K)
        np.add.at(cs, yp, Xp * cn[keep, None]); np.add.at(cc, yp, cn[keep])
        cm = cs / np.maximum(cc, 1)[:, None]
        res['NCM'] = float(np.mean(((X[te] ** 2).sum(1)[:, None] - 2 * X[te] @ cm.T + (cm ** 2).sum(1)[None]).argmin(1) == y[te]))
        res['cells-1NN'] = float(np.mean(yp[((X[te] ** 2).sum(1)[:, None] - 2 * X[te] @ Xp.T + (Xp ** 2).sum(1)[None]).argmin(1)] == y[te]))
        try:
            res['GBG++'] = gb_acc(uniq(gbgpp(Xp, yp)), Xp, yp, X[te], y[te])
        except Exception as e:
            res['GBG++'] = None
        res['GBclassic'] = gb_acc(uniq(gbclassic(Xp, yp, seed=rep)), Xp, yp, X[te], y[te])
        for nm, frac in (('CGB_3', 1 / 3), ('CGB_10', 1 / 10)):
            try:
                lam = lam_for_target(Xp, yp, max(K, int(len(Xp) * frac)), steps=10)
                m = CGB(lam=lam, k=min(10, len(Xp) - 1)).fit(Xp, yp)
                res[nm] = gb_acc(m.labels_, Xp, yp, X[te], y[te]); res[nm + '_balls'] = int(m.n_balls_)
            except Exception:
                res[nm] = None
        res['CGB'] = res['CGB_3']
        out.append(res)
    return out


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    X, y, name = load(did, 20000)
    res = Parallel(n_jobs=nj)(delayed(one)(X, y, r, M) for r in range(reps) for M in (25, 100))
    with open(outp, 'w') as f:
        for rr in res:
            for r in rr:
                r.update(did=int(did), dataset=name, n=len(X), d=X.shape[1])
                f.write(json.dumps(r) + '\n')
    print('done', name)
