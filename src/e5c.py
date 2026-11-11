"""E5c (= E5b + CGB-sig): reproducibility of differentially private granular-ball releases.
Same mechanism as E5 (public anchors -> per (anchor,class) noisy counts and sums, analytic Gaussian
mechanism, (eps,1e-5)-DP, post-processing). For every eps we draw TWO independent releases from the
same private data and fit each model on each release. We report accuracy (mean of the two releases)
and the disagreement rate between the two models' test predictions. One fixed prediction rule
(density rule) is used for every GB method; nothing is selected on the test set.
usage: python e5b.py <openml_id> <out.jsonl> [reps] [njobs]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load, uniq, balls, predict, lam_for_target
from cgb import CGB
from baselines import gbgpp, gbclassic
from e5_dp import gauss_sigma

EPSS = (0.25, 0.5, 1.0, 2.0, 4.0)
DELTA = 1e-5


def release(cnt, S, sig, rng, K, d):
    G = len(cnt)
    cn = cnt + sig * rng.standard_normal(G); Sn = S + sig * rng.standard_normal(S.shape)
    tau = max(3.0, 3 * sig); keep = cn > tau
    if keep.sum() < 2 * K:
        keep = np.isin(np.arange(G), np.argsort(-cn)[:2 * K])
    Xp = np.clip(Sn[keep] * np.sqrt(d) / cn[keep, None], 0, 1); yp = (np.arange(G) % K)[keep]
    return Xp, yp, cn[keep]


def models(Xp, yp, cw, Xte, K, sig=0.0):
    out = {}
    cs = np.zeros((K, Xp.shape[1])); cc = np.zeros(K)
    np.add.at(cs, yp, Xp * cw[:, None]); np.add.at(cc, yp, cw)
    cm = cs / np.maximum(cc, 1)[:, None]
    out['NCM'] = ((Xte[:, None] - cm[None]) ** 2).sum(2).argmin(1)
    out['cells-1NN'] = yp[((Xte ** 2).sum(1)[:, None] - 2 * Xte @ Xp.T + (Xp ** 2).sum(1)[None]).argmin(1)]
    def gbpred(mem):
        C, Rr, yb, dens = balls(uniq(mem), Xp, yp); return predict(C, Rr, yb, dens, Xte, 'den')
    try:
        out['GBG++'] = gbpred(gbgpp(Xp, yp))
    except Exception:
        out['GBG++'] = None
    out['GBclassic'] = gbpred(gbclassic(Xp, yp, seed=0))
    if len(Xp) > 12:
        lam = lam_for_target(Xp, yp, max(K, int(len(Xp) / 3)), steps=10)
        out['CGB'] = gbpred(CGB(lam=lam, k=min(10, len(Xp) - 1)).fit(Xp, yp).labels_)
    else:
        out['CGB'] = gbpred(np.arange(len(Xp)))
    # CGB-sig: lambda set from the public noise scale, fixed a priori (no tuning):
    # per-coordinate noise of a released cell mean is sig*sqrt(d)/count, its norm sig*d/count.
    lam_s = float(np.median(sig * Xp.shape[1] / np.maximum(cw, 1.0))) if sig > 0 else 0.0
    if lam_s > 0 and len(Xp) > 12:
        out['CGB-sig'] = gbpred(CGB(lam=lam_s, k=min(10, len(Xp) - 1)).fit(Xp, yp).labels_)
    else:
        out['CGB-sig'] = gbpred(np.arange(len(Xp)))
    return out


def one(X, y, rep, M):
    rng = np.random.default_rng(rep); n, d = X.shape; K = int(y.max() + 1)
    p = rng.permutation(n); npub = n // 10
    pub, prv, te = p[:npub], p[npub:npub + int(0.65 * n)], p[npub + int(0.65 * n):]
    A = KMeans(min(M, npub), n_init=1, random_state=rep).fit(X[pub]).cluster_centers_
    a = ((X[prv] ** 2).sum(1)[:, None] - 2 * X[prv] @ A.T + (A ** 2).sum(1)[None]).argmin(1)
    cell = a * K + y[prv]; G = len(A) * K
    cnt = np.bincount(cell, minlength=G).astype(float)
    S = np.zeros((G, d)); np.add.at(S, cell, X[prv] / np.sqrt(d))
    res = []
    for eps in EPSS:
        sig = gauss_sigma(eps, DELTA, np.sqrt(2.0))
        P = []
        for draw in range(2):
            r2 = np.random.default_rng(10 ** 6 + 1000 * rep + 10 * draw + int(10 * eps))
            Xp, yp, cw = release(cnt, S, sig, r2, K, d)
            P.append(models(Xp, yp, cw, X[te], K, sig))
        row = dict(M=M, rep=rep, eps=eps, sigma=sig)
        for m in P[0]:
            if P[0][m] is None or P[1][m] is None:
                continue
            row[m + '_acc'] = float(np.mean([np.mean(P[k][m] == y[te]) for k in range(2)]))
            row[m + '_disagree'] = float(np.mean(P[0][m] != P[1][m]))
        res.append(row)
    return res


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    X, y, name = load(did, 20000)
    res = Parallel(n_jobs=nj)(delayed(one)(X, y, r, M) for r in range(reps) for M in (25, 100))
    with open(outp, 'w') as f:
        for rr in res:
            for r in rr:
                r.update(did=int(did), dataset=name, n=len(X), d=X.shape[1]); f.write(json.dumps(r) + '\n')
    print('done', name)
