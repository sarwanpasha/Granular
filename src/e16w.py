"""E16: cost at scale, all generators on one machine (greedy generators on CPU, CGB radius graph on GPU).
Per data set: stratified 20% test (at most 20000 points, seed 0); training sizes 12500 .. full (nested prefixes
of a seeded permutation). CGB: radius graph (CPU) + dual FISTA on the GPU in float32, tol 1e-5, <= 5000 iters;
lambda matched (12-step bisection) to the GBG++ ball count at n=25000 and then kept fixed for all n.
Greedy runs are killed after 3600 s. At the largest n <= 100000: sub95 (one 95% subsample, ARI on kept points)
for every method, and order (one row permutation) for the greedy generators.
usage: python e16w.py <openml_id> <out.jsonl>   (CUDA_VISIBLE_DEVICES=1)
"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np, torch
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e9g
from baselines import gbgpp, gbclassic, rggb_s, scorgbg_s

e9g.f8 = torch.float32
G = {'GBG++': lambda X, y: gbgpp(X, y), 'GBclassic': lambda X, y: gbclassic(X, y, seed=0),
     'RGGB-s': lambda X, y: rggb_s(X, y, seed=0), 'ScOrGBG-s': lambda X, y: scorgbg_s(X, y, seed=0)}
SIZES = [12500, 25000, 50000, 100000, 200000, 400000]
TMAX = 3600


def _w(name, X, y, q):
    t = time.time(); lab = np.asarray(G[name](X, y)); q.put((lab, time.time() - t))


def greedy(name, X, y):
    q = mp.Queue(); p = mp.Process(target=_w, args=(name, X, y, q)); p.start()
    try:
        lab, sec = q.get(timeout=TMAX)
    except Exception:
        p.terminate(); p.join(); return None, None
    p.join(); return np.unique(lab, return_inverse=True)[1], sec


def graph(X, y):
    t = time.time()
    r = float(np.median(NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1]))
    i, j, w = e9g.radius_graph(X, y, r)
    return (i, j, w), time.time() - t


def cgb(X, y, lam, gr=None):
    gsec = 0.0
    if gr is None: gr, gsec = graph(X, y)
    i, j, w = gr; t = time.time()
    U, fused, it = e9g.solve(X, i, j, lam * w, 1e-5, 5000); torch.cuda.synchronize()
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(len(X), len(X)))
    lab = connected_components(A, directed=False)[1]
    return lab, gsec, time.time() - t, int(it), len(i)


def acc(lab, X, y, Xt, yt):
    nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, X); C /= cnt[:, None]
    R = np.zeros(nb); np.add.at(R, lab, np.linalg.norm(X - C[lab], axis=1)); R /= cnt
    yb = np.array([np.bincount(y[lab == b]).argmax() for b in range(nb)])
    out = {}
    for rule in ('rad', 'den'):
        pr = np.empty(len(Xt), int); Ct = torch.tensor(C, device='cuda')
        sub = torch.tensor(R if rule == 'rad' else cnt / cnt.sum(), device='cuda')
        for s in range(0, len(Xt), 4096):
            Z = torch.tensor(Xt[s:s + 4096], device='cuda')
            pr[s:s + 4096] = yb[torch.argmin(torch.cdist(Z, Ct) - sub[None], 1).cpu().numpy()]
        out[rule] = float(np.mean(pr == yt))
    return out


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    X, y, name = e9g.load(did, 10 ** 9)
    rng = np.random.default_rng(0); te = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]; te.append(rng.choice(ic, max(1, int(round(min(0.2, 20000 / len(y)) * len(ic)))), replace=False))
    te = np.sort(np.concatenate(te)); tr = np.setdiff1d(np.arange(len(y)), te)
    Xt, yt = X[te], y[te]; perm = tr[rng.permutation(len(tr))]
    sizes = [s for s in SIZES if s < len(perm)] + [len(perm)]
    fo = open(outp, 'a')
    def emit(d):
        d.update(did=int(did), dataset=name); fo.write(json.dumps(d) + '\n'); fo.flush(); print(json.dumps(d), flush=True)
    X0, y0 = X[perm[:25000]], y[perm[:25000]]
    tgt = int(greedy('GBG++', X0, y0)[0].max() + 1); gr0, _ = graph(X0, y0)
    a, b = np.log(1e-3), np.log(300.0); best = None
    for _ in range(12):
        m = 0.5 * (a + b); nb = int(cgb(X0, y0, float(np.exp(m)), gr0)[0].max() + 1)
        if best is None or abs(nb - tgt) < abs(best[1] - tgt): best = (float(np.exp(m)), nb)
        if nb == tgt: break
        if nb > tgt: a = m
        else: b = m
    lam = best[0]; emit(dict(kind='lambda', lam=lam, target=tgt, got=best[1]))
    stab_n = max(s for s in sizes if s <= 100000)
    for n in sizes:
        Xn, yn = X[perm[:n]], y[perm[:n]]
        for g in G:
            if g == 'RGGB-s' and n >= int(os.environ.get('RGGB_MAXN', 10 ** 12)):
                emit(dict(kind='fit', method=g, n=n, sec=None, timeout=None, skipped='memory')); continue
            lab, sec = greedy(g, Xn, yn)
            d = dict(kind='fit', method=g, n=n, sec=sec, timeout=lab is None)
            if lab is not None:
                d.update(n_balls=int(lab.max() + 1), **{'acc_' + k: v for k, v in acc(lab, Xn, yn, Xt, yt).items()})
                if n == stab_n:
                    k = np.sort(np.random.default_rng(1).choice(n, int(0.95 * n), replace=False))
                    l2, _ = greedy(g, Xn[k], yn[k]); d['sub95'] = None if l2 is None else float(1 - ARI(lab[k], l2))
                    p = np.random.default_rng(2).permutation(n); l3, _ = greedy(g, Xn[p], yn[p])
                    if l3 is not None:
                        m3 = np.empty(n, int); m3[p] = l3; d['order'] = float(1 - ARI(lab, m3))
            emit(d)
        torch.cuda.reset_peak_memory_stats()
        lab, gsec, ssec, it, ne = cgb(Xn, yn, lam)
        d = dict(kind='fit', method='CGB-r', n=n, sec=gsec + ssec, graph_sec=gsec, solve_sec=ssec, iters=it, edges=ne,
                 n_balls=int(lab.max() + 1), gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                 **{'acc_' + k: v for k, v in acc(lab, Xn, yn, Xt, yt).items()})
        if n == stab_n:
            k = np.sort(np.random.default_rng(1).choice(n, int(0.95 * n), replace=False))
            d['sub95'] = float(1 - ARI(lab[k], cgb(Xn[k], yn[k], lam)[0]))
            p = np.random.default_rng(2).permutation(n); l3 = cgb(Xn[p], yn[p], lam)[0]
            m3 = np.empty(n, int); m3[p] = l3; d['order'] = float(1 - ARI(lab, m3))
        emit(d)
