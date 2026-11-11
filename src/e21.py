"""E21: E16 large-scale runs with lambda chosen by the E19d rule instead of matching the GBG++ ball count.
Same data, test split and nested training prefixes as e16w.py. Radius-graph CGB, GPU dual FISTA in
float32, tol 1e-5, <= 5000 iterations (as in E16).
 mode cv : inner 3-fold stratified CV (random_state=1) on the first 25000 training points for ONE lambda of the
           E19d grid {0.003, ..., 1000, inf}; inf = every same-label edge of the radius graph fused.
 mode fit: read all cv rows of the data set, choose (lambda, rule) with the highest inner accuracy (ties -> larger
           lambda), fit at training size n, report test accuracy (chosen rule and both rules), balls, time;
           with stab=1 also sub95 (one 95% subsample) and order (one permutation), seeds as in E16.
usage: python e21.py cv  <openml_id> <lam_index> <out.jsonl>
       python e21.py fit <openml_id> <n> <stab 0|1> <cv.jsonl glob> <out.jsonl>
"""
import sys, os, json, time, glob
import numpy as np, torch
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e9g as e10g
from baselines import gbgpp

e10g.f8 = torch.float32
GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, float('inf')]
SIZES = [12500, 25000, 50000, 100000, 200000, 400000]


def data(did):
    X, y, name = e10g.load(did, 10 ** 9)
    rng = np.random.default_rng(0); te = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]; te.append(rng.choice(ic, max(1, int(round(min(0.2, 20000 / len(y)) * len(ic)))), replace=False))
    te = np.sort(np.concatenate(te)); tr = np.setdiff1d(np.arange(len(y)), te)
    return X, y, name, X[te], y[te], tr[rng.permutation(len(tr))]


def graph(X, y):
    r = float(np.median(NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1]))
    return e10g.radius_graph(X, y, r)


def labels(X, y, lam):
    t = time.time(); i, j, w = graph(X, y); gsec = time.time() - t; t = time.time(); it = 0
    if np.isinf(lam):
        A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(len(X), len(X)))
    else:
        U, fused, it = e10g.solve(X, i, j, lam * w, 1e-5, 5000); torch.cuda.synchronize()
        A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(len(X), len(X)))
    lab = connected_components(A, directed=False)[1]
    return lab, gsec, time.time() - t, int(it), len(i)


def acc(lab, X, y, Xt, yt):
    nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, X); C /= cnt[:, None]
    R = np.zeros(nb); np.add.at(R, lab, np.linalg.norm(X - C[lab], axis=1)); R /= cnt
    yb = np.array([np.bincount(y[lab == b]).argmax() for b in range(nb)])
    out = {}; Ct = torch.tensor(C, device='cuda')
    for rule in ('rad', 'den'):
        sub = torch.tensor(R if rule == 'rad' else cnt / cnt.sum(), device='cuda'); pr = np.empty(len(Xt), int)
        for s in range(0, len(Xt), 4096):
            Z = torch.tensor(Xt[s:s + 4096], device='cuda')
            pr[s:s + 4096] = yb[torch.argmin(torch.cdist(Z, Ct) - sub[None], 1).cpu().numpy()]
        out[rule] = float(np.mean(pr == yt))
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'cv':
        did, li, outp = sys.argv[2], int(sys.argv[3]), sys.argv[4]; lam = GRID[li]
        X, y, name, Xt, yt, perm = data(did); p = perm[:25000]; X0, y0 = X[p], y[p]
        a = {'rad': [], 'den': []}; nbs = []; t0 = time.time()
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=1).split(X0, y0):
            lab = labels(X0[tr], y0[tr], lam)[0]; nbs.append(int(lab.max() + 1))
            s = acc(lab, X0[tr], y0[tr], X0[te], y0[te])
            for k in a: a[k].append(s[k])
        r = dict(kind='cv', did=int(did), dataset=name, lam=lam, lam_index=li, acc_rad=float(np.mean(a['rad'])),
                 acc_den=float(np.mean(a['den'])), n_balls_folds=nbs, sec=time.time() - t0)
        open(outp, 'w').write(json.dumps(r) + '\n'); print(json.dumps(r), flush=True)
    else:
        did, n, stab, cvg, outp = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
        cv = [json.loads(open(f).readline()) for f in sorted(glob.glob(cvg))]
        assert len(cv) == len(GRID), len(cv)
        cand = [(round(c['acc_' + rule], 12), c['lam'], rule) for c in cv for rule in ('rad', 'den')]
        best = max(cand, key=lambda z: (z[0], z[1])); lam, rule = best[1], best[2]
        X, y, name, Xt, yt, perm = data(did); fo = open(outp, 'a')
        def emit(d):
            d.update(did=int(did), dataset=name, gpu=torch.cuda.get_device_name(0))
            fo.write(json.dumps(d) + '\n'); fo.flush(); print(json.dumps(d), flush=True)
        Xc, yc = X[perm[:12500]], y[perm[:12500]]; lc = np.unique(gbgpp(Xc, yc), return_inverse=True)[1]
        emit(dict(kind='check', method='GBG++', n=12500, n_balls=int(lc.max() + 1), **{'acc_' + k: v for k, v in acc(lc, Xc, yc, Xt, yt).items()}))
        n = min(n, len(perm)); Xn, yn = X[perm[:n]], y[perm[:n]]
        torch.cuda.reset_peak_memory_stats()
        lab, gsec, ssec, it, ne = labels(Xn, yn, lam); a = acc(lab, Xn, yn, Xt, yt)
        d = dict(kind='fit', method='CGB-r-cv', n=n, lam=lam, rule=rule, inner_acc=best[0], sec=gsec + ssec, graph_sec=gsec,
                 solve_sec=ssec, iters=it, edges=ne, n_balls=int(lab.max() + 1), acc=a[rule], acc_rad=a['rad'], acc_den=a['den'],
                 gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        if stab:
            k = np.sort(np.random.default_rng(1).choice(n, int(0.95 * n), replace=False))
            d['sub95'] = float(1 - ARI(lab[k], labels(Xn[k], yn[k], lam)[0]))
            pp = np.random.default_rng(2).permutation(n); m3 = np.empty(n, int); m3[pp] = labels(Xn[pp], yn[pp], lam)[0]
            d['order'] = float(1 - ARI(lab, m3))
        emit(d)
