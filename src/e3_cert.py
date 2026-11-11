"""E3: certified stability radius (T3) vs empirical breaking radius, radius-graph CGB (theory setting).
usage: python e3_cert.py <openml_id> <out.jsonl> [nmax] [njobs]
"""
import sys, os, json, time
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load
from cgb import incidence, solve_cfuse
from sklearn.metrics import adjusted_rand_score as ARI


def radius_graph(X, y, r):
    nn = NearestNeighbors(radius=r).fit(X)
    A = nn.radius_neighbors_graph(X, mode='distance')
    A = sp.triu(A, k=1).tocoo()
    i, j, d = A.row, A.col, A.data
    t = np.clip(1 - (d / r) ** 2, 0, None); w = t * t
    m = (w > 0) & (y[i] == y[j])
    o = np.lexsort((j[m], i[m]))
    return i[m][o], j[m][o], w[m][o]


def fit(X, y, lam, r, tol=1e-10):
    i, j, w = radius_graph(X, y, r)
    D = incidence(i, j, len(X)); gam = lam * w
    U, Lam, info = solve_cfuse(X, D, gam, tol, 100000)
    nu = 1 / info['L']; Z = Lam + nu * (D @ U)
    zn = np.linalg.norm(Z, axis=1); fused = zn <= gam
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(len(X), len(X)))
    _, lab = connected_components(A, directed=False)
    return dict(lab=lab, U=U, Lam=Lam, gam=gam, fused=fused, D=D, L=info['L'], i=i, j=j, gap=info['gap'])


def certificate(f, lam, r, n):
    D = f['D']; U = f['U']; lab = f['lab']; i, j = f['i'], f['j']
    dU = np.linalg.norm(D @ U, axis=1)
    cross = lab[i] != lab[j]
    sep = dU[cross].min() if cross.any() else np.inf
    sl = f['gam'] - np.linalg.norm(f['Lam'], axis=1)
    intra = ~cross
    nb = lab.max() + 1
    cand = np.unique(np.r_[0.0, sl[intra & (sl > 0)]])
    def ok(t):
        m = intra & (sl >= t)
        A = sp.csr_matrix((np.ones(m.sum()), (i[m], j[m])), shape=(n, n))
        k, cl = connected_components(A, directed=False)
        return k == nb
    lo, hi = 0, len(cand) - 1
    if not ok(cand[0]):
        slack = 0.0
    else:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(cand[mid]): lo = mid
            else: hi = mid - 1
        slack = float(cand[lo]) if nb < n else np.inf
    deg = np.bincount(np.r_[i, j], minlength=n).max()
    Lphi = 8 / (3 * np.sqrt(3) * r)
    kappa = 1 + 2 * lam * Lphi * deg
    Dop = np.sqrt(f['L'])
    nC = np.bincount(lab).max()
    rho = min(sep / (Dop * kappa), slack / (nC * (1 + kappa) + 2 * lam * Lphi))
    return dict(sep=float(sep), slack=float(slack), kappa=float(kappa), deg=int(deg), Dop=float(Dop), nCmax=int(nC), rho=float(max(rho, 0)))


def breaking_radius(X, y, lam, r, lab0, seed, steps=14):
    rng = np.random.default_rng(seed); E = rng.standard_normal(X.shape); E /= np.linalg.norm(E)
    lo, hi = 0.0, 10.0
    if ARI(lab0, fit(X + hi * E, y, lam, r, 1e-9)['lab']) == 1.0:
        return float('inf')
    for _ in range(steps):
        mid = np.sqrt(max(lo, 1e-8) * hi) if lo > 0 else hi / 4
        same = ARI(lab0, fit(X + mid * E, y, lam, r, 1e-9)['lab']) == 1.0
        if same: lo = mid
        else: hi = mid
        if lo > 0 and hi / lo < 1.05: break
    return float(lo)


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    X, y, name = load(did, nmax)
    nn = NearestNeighbors(n_neighbors=11).fit(X); dk = nn.kneighbors(X)[0][:, -1]
    r = 2.0 * np.median(dk)
    rows = []
    def lam_for(frac):
        a, b = np.log(1e-3), np.log(300.0)
        for _ in range(12):
            mid = 0.5 * (a + b); nb = fit(X, y, float(np.exp(mid)), r, 1e-8)['lab'].max() + 1
            if nb > frac * len(X): a = mid
            else: b = mid
        return float(np.exp(0.5 * (a + b)))
    lams = Parallel(n_jobs=3)(delayed(lam_for)(fr) for fr in (1 / 3, 1 / 10, 1 / 30))
    for lam in lams:
        f = fit(X, y, lam, r)
        c = certificate(f, lam, r, len(X))
        br = Parallel(n_jobs=nj)(delayed(breaking_radius)(X, y, lam, r, f['lab'], s) for s in range(nj))
        fin = [b for b in br if np.isfinite(b)]
        row = dict(did=int(did), dataset=name, n=len(X), d=X.shape[1], lam=lam, r=float(r), n_balls=int(f['lab'].max() + 1),
                   gap=float(f['gap']), emp_break_min=float(min(fin)) if fin else None,
                   emp_break_med=float(np.median(fin)) if fin else None, **c)
        row['ratio'] = (row['emp_break_min'] / row['rho']) if (row['emp_break_min'] and row['rho'] > 0) else None
        if c['rho'] > 0:
            E = np.random.default_rng(99).standard_normal(X.shape); E *= 0.99 * c['rho'] / np.linalg.norm(E)
            row['cert_holds'] = bool(ARI(f['lab'], fit(X + E, y, lam, r, 1e-11)['lab']) == 1.0)
        print(json.dumps(row), flush=True); rows.append(row)
    with open(outp, 'w') as fo:
        for rr in rows:
            fo.write(json.dumps(rr) + '\n')
