"""E19d (= E19c plus lambda = infinity, the end of the lambda path, as the last grid value). E19: choosing lambda without a target ball count. Rule: inner 3-fold stratified CV on the training data over
the grid {0.003, ..., 100, 300, 1000} (E19c: grid extended again after 10 of 51 data sets chose 100 in E19b) jointly with the prediction rule (radius / density); ties go
to the larger lambda (fewer balls). Nested evaluation: the same outer 5-fold split as the benchmark
(StratifiedKFold(5, shuffle=True, random_state=0)); for every outer fold lambda and rule are chosen on the
training part only, and test accuracy is recorded. GBG++ is evaluated in the same nested way (its rule chosen
by inner CV), so both sides are free of test-set selection. Finally lambda* is chosen by the same inner CV on
the full data and CGB at lambda* is run through bench.evaluate (order, jitter, sub95, drift).
usage: python e19d.py <openml_id> <out.jsonl> <nmax> [njobs]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from bench import balls, predict, uniq
from bench_r import gen_r, gen_any
from e2c_ablate import fit_variant

GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, float('inf')]
RULES = ('rad', 'den')


def fitlab(X, y, lam):
    if np.isinf(lam):  # end of the lambda path: every same-label edge of the radius graph fused
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        from sklearn.neighbors import NearestNeighbors
        from e3_cert import radius_graph
        r = np.median(NearestNeighbors(n_neighbors=min(11, len(X))).fit(X).kneighbors(X)[0][:, -1])
        i, j, w = radius_graph(X, y, r)
        A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(len(X), len(X)))
        return connected_components(A, directed=False)[1]
    return uniq(fit_variant(X, y, lam, 'radius')[0])


def gen_inf(method, X, y, seed=0, lam=None):
    if lam is not None and np.isinf(lam):
        lab = fitlab(X, y, lam); nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
        C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, X); C /= cnt[:, None]
        return lab, C[lab]
    return gen_any(method, X, y, seed, lam)


def gbgpp_lab(X, y):
    return bench.gen('GBG++', X, y, 0, None)[0]


def score(lab, Xtr, ytr, Xte, yte):
    C, Rr, yb, dens = balls(lab, Xtr, ytr)
    return {r: float(np.mean(predict(C, Rr, yb, dens, Xte, r) == yte)) for r in RULES}, len(C)


def inner_cv(X, y, labf_list):
    """labf_list: list of (key, f(Xtr, ytr) -> labels). Returns mean inner accuracy per (key, rule)."""
    acc = {}
    nmin = np.bincount(y).min(); k = 3 if nmin >= 3 else 2
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=1).split(X, y):
        for key, f in labf_list:
            s, _ = score(f(X[tr], y[tr]), X[tr], y[tr], X[te], y[te])
            for r in RULES: acc.setdefault((key, r), []).append(s[r])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def select_lam(X, y):
    a = inner_cv(X, y, [(lam, (lambda Xt, yt, lam=lam: fitlab(Xt, yt, lam))) for lam in GRID])
    best = max(a.items(), key=lambda kv: (round(kv[1], 12), kv[0][0]))
    return best[0][0], best[0][1], best[1]


def outer_fold(X, y, tr, te):
    t0 = time.time()
    lam, rule, inner = select_lam(X[tr], y[tr])
    s, nb = score(fitlab(X[tr], y[tr], lam), X[tr], y[tr], X[te], y[te])
    g = inner_cv(X[tr], y[tr], [('g', gbgpp_lab)]); grule = max(RULES, key=lambda r: g[('g', r)])
    gs, gnb = score(gbgpp_lab(X[tr], y[tr]), X[tr], y[tr], X[te], y[te])
    return dict(lam=lam, rule=rule, inner=inner, acc=s[rule], n_balls=nb,
                g_rule=grule, g_acc=gs[grule], g_n_balls=gnb, sec=time.time() - t0)


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    X, y, name = bench.load(did, nmax)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    res = Parallel(n_jobs=nj)(delayed(outer_fold)(X, y, tr, te) for tr, te in folds)
    lam_star, rule_star, _ = select_lam(X, y)
    bench.gen = gen_inf
    st = bench.evaluate('CGB-r', X, y, lam_star, 11)
    out = dict(did=int(did), dataset=name, n=len(X), folds=res, lam_star=lam_star, rule_star=rule_star,
               acc_cv=float(np.mean([r['acc'] for r in res])), g_acc_cv=float(np.mean([r['g_acc'] for r in res])),
               stab=st)
    with open(outp, 'w') as fo:
        fo.write(json.dumps(out) + '\n')
    print('done', did, name, len(X), lam_star, out['acc_cv'], out['g_acc_cv'], flush=True)
