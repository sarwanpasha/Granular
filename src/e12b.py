"""E12b (= E12 with canonical ball ordering in every learner): does generator stability carry through to downstream granular-ball learners?
For each data set and 5 random 70/30 train/test splits, each generator builds balls on the training split;
four downstream learners are trained on the balls only:
  GB-NB   nearest ball by ||x-c|| - rho            GB-kNN  size-weighted vote of the 3 nearest balls
  GB-SVM  RBF SVC on ball centres (sample_weight = ball size, C=1, gamma='scale')
  GB-LR   logistic regression on ball centres (sample_weight = ball size)
The generator is then re-run 3 times after a row permutation, 3 times with a new seed and 3 times on a
random 95% subsample of the training split, the learners are retrained, and we record the fraction of test
points whose prediction changes (disagreement with the reference run). CGB-r uses the lambda matched to
GBG++ in the benchmark (results/bench_r), fixed for all reruns.
usage: python e12b.py <openml_id> <out.jsonl> <nmax> [njobs]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from bench import uniq, balls, predict
import bench_r

GENS = ('GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s', 'CGB-r')


def gen(m, X, y, seed, lam):
    if m == 'CGB-r':
        return uniq(bench_r.gen_r('CGB-r', X, y, 0, lam)[0])
    return uniq(bench.gen(m, X, y, seed, None)[0])


def learners(lab, X, y, Xt, K):
    C, Rr, yb, dens = balls(lab, X, y); cnt = np.bincount(lab).astype(float)
    o = np.lexsort(np.c_[C, Rr, yb].T[::-1])  # E12b: canonical ball order (by centre, radius, label) so ties break identically
    C, Rr, yb, dens, cnt = C[o], Rr[o], yb[o], dens[o], cnt[o]
    out = {'GB-NB': predict(C, Rr, yb, dens, Xt, 'rad')}
    d = np.sqrt(np.maximum((Xt ** 2).sum(1)[:, None] - 2 * Xt @ C.T + (C ** 2).sum(1)[None], 0)) - Rr[None]
    k = min(3, len(C)); idx = np.argpartition(d, k - 1, axis=1)[:, :k]; V = np.zeros((len(Xt), K))
    for j in range(k):
        np.add.at(V, (np.arange(len(Xt)), yb[idx[:, j]]), cnt[idx[:, j]])
    out['GB-kNN'] = V.argmax(1)
    if len(np.unique(yb)) > 1:
        out['GB-SVM'] = SVC(C=1.0, gamma='scale').fit(C, yb, sample_weight=cnt).predict(Xt)
        out['GB-LR'] = LogisticRegression(max_iter=2000).fit(C, yb, sample_weight=cnt).predict(Xt)
    else:
        out['GB-SVM'] = out['GB-LR'] = np.full(len(Xt), yb[0])
    return out, len(C)


def one(X, y, split, lam, did):
    rng = np.random.default_rng(split); n = len(X); p = rng.permutation(n)
    tr, te = p[:int(0.7 * n)], p[int(0.7 * n):]; Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]; K = int(y.max() + 1)
    rows = []
    for m in GENS:
        t0 = time.time(); ref, nb = learners(gen(m, Xtr, ytr, split, lam), Xtr, ytr, Xte, K); sec = time.time() - t0
        chg = {v: {l: [] for l in ref} for v in ('order', 'seed', 'sub95')}
        for r in range(3):
            pr = np.random.default_rng(100 * split + r).permutation(len(tr))
            o, _ = learners(gen(m, Xtr[pr], ytr[pr], split, lam), Xtr[pr], ytr[pr], Xte, K)
            for l in ref: chg['order'][l].append(np.mean(o[l] != ref[l]))
            o, _ = learners(gen(m, Xtr, ytr, split + 1000 * (r + 1), lam), Xtr, ytr, Xte, K)
            for l in ref: chg['seed'][l].append(np.mean(o[l] != ref[l]))
            kk = np.sort(np.random.default_rng(5000 + 100 * split + r).choice(len(tr), int(0.95 * len(tr)), replace=False))
            o, _ = learners(gen(m, Xtr[kk], ytr[kk], split, lam), Xtr[kk], ytr[kk], Xte, K)
            for l in ref: chg['sub95'][l].append(np.mean(o[l] != ref[l]))
        row = dict(did=did, split=split, gen=m, n_balls=nb, sec_ref=sec, lam=lam if m == 'CGB-r' else None)
        for l in ref:
            row[f'acc_{l}'] = float(np.mean(ref[l] == yte))
            for v in chg:
                row[f'chg_{v}_{l}'] = float(np.mean(chg[v][l]))
        rows.append(row)
    return rows


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    X, y, name = bench.load(did, nmax)
    lam = [json.loads(l) for l in open(f'results/bench_r/r_{did}.jsonl') if json.loads(l)['matched_to'] == 'GBG++'][0]['lam']
    res = Parallel(n_jobs=nj)(delayed(one)(X, y, s, lam, int(did)) for s in range(5))
    with open(outp, 'w') as f:
        for rr in res:
            for r in rr:
                r.update(dataset=name, n=len(X), d=X.shape[1]); f.write(json.dumps(r) + '\n')
    print('done', name)
