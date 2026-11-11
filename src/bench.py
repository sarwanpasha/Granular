"""E1/E2 benchmark: stability at matched granularity + accuracy, many OpenML datasets.
usage: python bench.py <openml_data_id> <out.jsonl> [nmax] [njobs]
"""
import sys, json, time, os
import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgb import CGB
from baselines import gbgpp, gbclassic, rggb_s, scorgbg_s

EPS = (0.01, 0.05)
R = 5


def load(did, nmax, seed=0):
    import pandas as pd
    d = fetch_openml(data_id=int(did), as_frame=True, parser='auto', data_home='data')
    df = d.data.copy()
    for c in df.columns:
        if str(df[c].dtype) in ('category', 'object', 'bool'):
            df[c] = df[c].astype('category')
    num = df.select_dtypes(include='number')
    cat = df.select_dtypes(exclude='number')
    num = num.fillna(num.median())
    parts = [num]
    if cat.shape[1]:
        cat = cat.apply(lambda s: s.cat.add_categories(['__na']).fillna('__na') if hasattr(s, 'cat') else s)
        parts.append(pd.get_dummies(cat, dtype=float))
    X = pd.concat(parts, axis=1).to_numpy(dtype=float)
    _, y = np.unique(np.asarray(d.target).astype(str), return_inverse=True)
    X = X[:, np.nanstd(X, 0) > 0]
    cnt = np.bincount(y); keep = cnt[y] >= 10; X, y = X[keep], y[keep]; _, y = np.unique(y, return_inverse=True)
    X = MinMaxScaler().fit_transform(X)
    _, first = np.unique(np.round(X, 10), axis=0, return_index=True)
    first = np.sort(first); X, y = X[first], y[first]
    if len(X) > nmax:
        rng = np.random.default_rng(seed)
        idx = []
        for c in np.unique(y):
            ic = np.where(y == c)[0]; k = max(1, int(round(nmax * len(ic) / len(y))))
            idx.append(rng.choice(ic, min(k, len(ic)), replace=False))
        idx = np.sort(np.concatenate(idx)); X, y = X[idx], y[idx]
    return X, y, d.details.get('name', str(did))


def uniq(m):
    m = np.asarray(m).copy(); neg = m < 0
    if neg.any():
        m[neg] = (m.max() + 1 if (~neg).any() else 0) + np.arange(neg.sum())
    return m


def gen(method, X, y, seed=0, lam=None):
    if method == 'CGB':
        m = CGB(lam=lam).fit(X, y); lab = m.labels_; Z = m.U_
    else:
        f = {'GBG++': lambda: gbgpp(X, y), 'GBclassic': lambda: gbclassic(X, y, seed=seed),
             'RGGB-s': lambda: rggb_s(X, y, seed=seed), 'ScOrGBG-s': lambda: scorgbg_s(X, y, seed=seed)}[method]
        lab = uniq(f()); Z = X
    lab = uniq(lab); nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, Z); C /= np.maximum(cnt, 1)[:, None]
    return lab, C[lab]


def lam_for_target(X, y, target, lo=1e-3, hi=10.0, steps=12):
    a, b = np.log(lo), np.log(hi); best = None
    for _ in range(steps):
        mid = 0.5 * (a + b); lam = float(np.exp(mid))
        nb = CGB(lam=lam, tol=1e-7).fit(X, y).n_balls_
        if best is None or abs(nb - target) < abs(best[1] - target):
            best = (lam, nb)
        if nb == target: break
        if nb > target: a = mid
        else: b = mid
    return best[0]


def balls(mem, X, y):
    mem = uniq(mem); nb = mem.max() + 1; cnt = np.bincount(mem, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, mem, X); C /= np.maximum(cnt, 1)[:, None]
    Rr = np.zeros(nb); np.add.at(Rr, mem, np.linalg.norm(X - C[mem], axis=1)); Rr /= np.maximum(cnt, 1)
    yb = np.array([np.bincount(y[mem == b]).argmax() for b in range(nb)])
    return C, Rr, yb, cnt / cnt.sum()


def predict(C, Rr, yb, dens, Xt, rule):
    out = np.empty(len(Xt), dtype=yb.dtype)
    for s in range(0, len(Xt), 1024):
        Z = Xt[s:s + 1024]
        d = np.sqrt(np.maximum((Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None], 0))
        d = d - (Rr[None] if rule == 'rad' else dens[None])
        out[s:s + 1024] = yb[np.argmin(d, 1)]
    return out


def evaluate(method, X, y, lam, seed0):
    rng = np.random.default_rng(seed0)
    t0 = time.time()
    n = len(X); out = {}
    ref, Zref = gen(method, X, y, 0, lam)
    out['n_balls'] = int(ref.max() + 1)
    out['gen_sec'] = time.time() - t0
    v = []
    for r in range(R):
        p = rng.permutation(n); m = np.empty(n, int); m[p] = gen(method, X[p], y[p], 0, lam)[0]
        v.append(1 - ARI(ref, m))
    out['order'] = float(np.mean(v))
    if method in ('GBclassic', 'RGGB-s', 'ScOrGBG-s'):
        out['seed'] = float(np.mean([1 - ARI(ref, gen(method, X, y, s + 100, lam)[0]) for s in range(R)]))
    for eps in EPS:
        v = []; dr = []
        for r in range(R):
            E = eps * rng.standard_normal(X.shape)
            mp, Zp = gen(method, X + E, y, 0, lam)
            v.append(1 - ARI(ref, mp)); dr.append(np.linalg.norm(Zp - Zref) / np.linalg.norm(E))
        out[f'jit{eps}'] = float(np.mean(v)); out[f'drift{eps}'] = float(np.mean(dr))
    v = []
    for r in range(R):
        keep = np.sort(rng.choice(n, int(0.95 * n), replace=False))
        v.append(1 - ARI(ref[keep], gen(method, X[keep], y[keep], 0, lam)[0]))
    out['sub95'] = float(np.mean(v))
    skf = StratifiedKFold(5, shuffle=True, random_state=0); acc = {'rad': [], 'den': []}; nbs = []
    for tr, te in skf.split(X, y):
        mem = gen(method, X[tr], y[tr], 0, lam)[0]
        C, Rr, yb, dens = balls(mem, X[tr], y[tr]); nbs.append(len(C))
        for rule in acc:
            acc[rule].append(np.mean(predict(C, Rr, yb, dens, X[te], rule) == y[te]))
    out['acc_rad'] = float(np.mean(acc['rad'])); out['acc_den'] = float(np.mean(acc['den']))
    out['acc'] = max(out['acc_rad'], out['acc_den'])
    out['cv_balls'] = float(np.mean(nbs))
    out['sec'] = time.time() - t0
    return out


def refs(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    o = {}
    for nm, clf in [('kNN5', KNeighborsClassifier(5)), ('SVM', SVC()), ('RF', RandomForestClassifier(200, random_state=0, n_jobs=1))]:
        o[nm] = float(cross_val_score(clf, X, y, cv=skf).mean())
    return o


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    X, y, name = load(did, nmax)
    meta = dict(did=int(did), dataset=name, n=len(X), d=X.shape[1], k=int(y.max() + 1))
    print(json.dumps(meta), flush=True)
    bases = ['GBG++', 'GBclassic', 'RGGB-s', 'ScOrGBG-s']
    nb = Parallel(n_jobs=nj)(delayed(lambda m: int(gen(m, X, y, 0)[0].max() + 1))(m) for m in bases)
    lams = Parallel(n_jobs=nj)(delayed(lam_for_target)(X, y, t) for t in nb)
    cfgs = [(m, None, None) for m in bases] + [('CGB', l, b) for l, b in zip(lams, bases)]
    res = Parallel(n_jobs=nj)(delayed(evaluate)(m, X, y, l, 7 + i) for i, (m, l, _) in enumerate(cfgs))
    with open(outp, 'w') as f:
        for (m, l, b), r in zip(cfgs, res):
            r.update(meta, method=m, lam=l, matched_to=b)
            f.write(json.dumps(r) + '\n')
        f.write(json.dumps(dict(meta, method='REF', **refs(X, y))) + '\n')
    print('done', name, flush=True)
