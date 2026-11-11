"""Convex Granular Balls (CGB): label-aware convex fusion granulation.

min_U 1/2 ||U - X||_F^2 + lam * sum_{(i,j) in E} w_ij ||u_i - u_j||_2
Solved by accelerated projected gradient (FISTA) on the dual.
Unique solution -> seed-free and order-invariant; prox map -> Lipschitz in X.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def smooth_cut(d, r):
    """Compactly supported, C^1 weight: (1 - (d/r)^2)_+^2 (Lipschitz in d)."""
    t = np.clip(1.0 - (d / r) ** 2, 0.0, None)
    return t * t


def build_graph(X, y=None, k=10, r=None, label_aware=True):
    """Symmetric kNN graph, canonical edge order (i<j, lexicographic).
    Weights use a smooth compact kernel with bandwidth r (default: 2x median kNN dist).
    Cross-label edges are removed when label_aware (=> pure balls)."""
    n = X.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = ind[:, 1:].ravel()
    dd = dist[:, 1:].ravel()
    i = np.minimum(rows, cols); j = np.maximum(rows, cols)
    keep = i != j
    i, j, dd = i[keep], j[keep], dd[keep]
    key = i.astype(np.int64) * n + j
    key, first = np.unique(key, return_index=True)  # canonical, order-free
    i, j, dd = i[first], j[first], dd[first]
    if r is None:
        r = 2.0 * np.median(dist[:, -1]) + 1e-12
    w = smooth_cut(dd, r)
    if label_aware and y is not None:
        w = w * (y[i] == y[j])
    m = w > 0
    return i[m], j[m], w[m], r


def incidence(i, j, n):
    m = len(i)
    data = np.concatenate([np.ones(m), -np.ones(m)])
    rr = np.concatenate([np.arange(m), np.arange(m)])
    cc = np.concatenate([i, j])
    return sp.csr_matrix((data, (rr, cc)), shape=(m, n))


def _lmax(D, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(D.shape[1]); v /= np.linalg.norm(v)
    L = None
    for _ in range(iters):
        u = D.T @ (D @ v)
        L = np.linalg.norm(u); v = u / (L + 1e-300)
    return 1.02 * L


def solve_cfuse(X, D, gam, tol=1e-7, max_iter=20000, Lam0=None, verbose=False):
    """Dual FISTA. Returns U, Lam, info.  ||Lam_l|| <= gam_l."""
    m = D.shape[0]
    L = _lmax(D) if m > 0 else 1.0
    nu = 1.0 / L
    DX = D @ X
    Lam = np.zeros((m, X.shape[1])) if Lam0 is None else Lam0.copy()
    Y = Lam.copy(); t = 1.0
    g = gam[:, None]

    def proj(Z):
        nz = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z * np.minimum(1.0, g / np.maximum(nz, 1e-300))

    gap = np.inf; it = 0
    for it in range(1, max_iter + 1):
        U = X - D.T @ Y
        Lnew = proj(Y + nu * (D @ U))
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Y = Lnew + ((t - 1) / tn) * (Lnew - Lam)
        Lam, t = Lnew, tn
        if it % 50 == 0 or it == max_iter:
            U = X - D.T @ Lam
            DU = D @ U
            P = 0.5 * np.sum((U - X) ** 2) + np.sum(gam * np.linalg.norm(DU, axis=1))
            Gd = np.sum(Lam * DX) - 0.5 * np.sum((D.T @ Lam) ** 2)
            gap = P - Gd
            if verbose:
                print(it, P, Gd, gap)
            if gap <= tol * max(1.0, abs(P)):
                break
            # adaptive restart
            if np.sum((Lnew - Y) * (Lam - Lnew)) > 0:
                Y = Lam.copy(); t = 1.0
    U = X - D.T @ Lam
    return U, Lam, dict(iters=it, gap=gap, L=L)


class CGB:
    """Convex Granular Ball generator."""

    def __init__(self, lam=1.0, k=10, r=None, label_aware=True, fuse_tol=1e-3,
                 tol=1e-8, max_iter=20000):
        self.lam, self.k, self.r = lam, k, r
        self.label_aware, self.fuse_tol = label_aware, fuse_tol
        self.tol, self.max_iter = tol, max_iter

    def fit(self, X, y=None, graph=None):
        X = np.asarray(X, float)
        n = X.shape[0]
        if graph is None:
            graph = build_graph(X, y, self.k, self.r, self.label_aware)
        i, j, w, r = graph
        self.r_ = r
        D = incidence(i, j, n)
        gam = self.lam * w
        U, Lam, info = solve_cfuse(X, D, gam, self.tol, self.max_iter)
        nu = 1.0 / info['L']
        Z = Lam + nu * (D @ U)
        zn = np.linalg.norm(Z, axis=1)
        # exact-zero primal fusion variable V_l = shrink(Z_l, gam_l)/nu
        dU = np.maximum(zn - gam, 0.0) / nu
        scale = np.sqrt(np.mean(np.sum((X - X.mean(0)) ** 2, 1))) + 1e-12
        fused = dU == 0.0
        self.zslack_ = gam - zn
        A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(n, n))
        nc, lab = connected_components(A, directed=False)
        self.U_, self.Lam_, self.info_ = U, Lam, info
        self.edges_ = (i, j, w); self.gam_ = gam; self.dU_ = dU; self.fused_ = fused
        self.labels_ = lab; self.n_balls_ = nc
        cnt = np.bincount(lab, minlength=nc)
        C = np.zeros((nc, X.shape[1])); np.add.at(C, lab, X); C /= cnt[:, None]
        rad = np.zeros(nc); np.add.at(rad, lab, np.linalg.norm(X - C[lab], axis=1)); rad /= cnt
        self.centers_, self.radii_, self.sizes_ = C, rad, cnt
        if y is not None:
            yb = np.zeros(nc, dtype=y.dtype)
            for b in range(nc):
                vals, c = np.unique(y[lab == b], return_counts=True); yb[b] = vals[np.argmax(c)]
            self.ball_y_ = yb
        self.scale_ = scale
        return self

    def certificate(self):
        """Stability margins: min dual slack on fused edges, min primal separation on unfused."""
        f = self.fused_
        slack = self.zslack_[f] if f.any() else np.array([np.inf])
        sep = self.dU_[~f].min() if (~f).any() else np.inf
        Dn = np.sqrt(self.info_['L'])
        rho = min(sep / 2.0, slack.min() / Dn)
        return dict(min_slack=float(slack.min()), sep=float(sep), rho=float(rho))

    def predict(self, Xt):
        d = np.linalg.norm(Xt[:, None, :] - self.centers_[None], axis=2) - self.radii_[None]
        return self.ball_y_[np.argmin(d, axis=1)]
