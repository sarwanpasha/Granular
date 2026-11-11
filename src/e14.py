"""E14: certified-radius variants on the E3 settings (same data subsample, lambda and r as results/e3).
 rho_paper : the radius as originally stated (spanning forest, n_C (1+kappa)).
 rho_tree  : corrected spanning-forest radius. Fused edges outside the routing forest may have to shrink
             their dual when their weight drops; that shrinkage is rerouted through the forest, which raises the
             demand change from (1+kappa)||dX|| to 2 kappa ||dX||:  slack / (n_C * 2 kappa + 2 lam L_phi).
 rho_er    : electrical-flow routing on the subgraph S_t of fused edges with slack >= t (t ranges over slack
             values for which S_t still connects every ball): per-edge flow ||f_l|| <= ||(D_S L_S^+)_l||_2 ||db||_F,
             so rho_er = max_t min_{l in S_t} slack_l / (||(D_S L_S^+)_l|| 2 kappa + 2 lam L_phi).
 For every variant the certificate requires min(sep / (||D|| kappa), feasibility radius).
 Also reported: ||dX||_F for rounding X to 6, 5 and 4 decimals, i.e. whether rounding is certified harmless.
usage: python e14.py <openml_id> <out.jsonl>
"""
import sys, os, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load
from e3_cert import fit, certificate
from sklearn.metrics import adjusted_rand_score as ARI


def rho_variants(f, lam, r, n):
    lab = f['lab']; i, j = f['i'], f['j']; D = f['D']; U = f['U']
    dU = np.linalg.norm(D @ U, axis=1); cross = lab[i] != lab[j]
    sep = dU[cross].min() if cross.any() else np.inf
    slack = f['gam'] - np.linalg.norm(f['Lam'], axis=1); intra = ~cross
    deg = np.bincount(np.r_[i, j], minlength=n).max()
    Lphi = 8 / (3 * np.sqrt(3) * r); kappa = 1 + 2 * lam * Lphi * deg; Dop = np.sqrt(f['L'])
    nb = lab.max() + 1; nC = np.bincount(lab).max()
    r_sep = sep / (Dop * kappa)

    def connects(m):
        A = sp.csr_matrix((np.ones(m.sum()), (i[m], j[m])), shape=(n, n))
        return connected_components(A, directed=False)[0] == nb

    cand = np.unique(slack[intra & (slack > 0)])
    # bottleneck slack (largest t keeping every ball connected)
    lo, hi, bott = 0, len(cand) - 1, 0.0
    if nb < n and len(cand) and connects(intra & (slack >= cand[0])):
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if connects(intra & (slack >= cand[mid])): lo = mid
            else: hi = mid - 1
        bott = float(cand[lo])
    elif nb == n:
        bott = np.inf
    rho_tree = min(r_sep, bott / (nC * 2 * kappa + 2 * lam * Lphi)) if np.isfinite(bott) else r_sep
    # electrical-flow radius over a grid of thresholds below the bottleneck
    best = 0.0
    if np.isfinite(bott) and bott > 0:
        ts = np.unique(np.quantile(cand[cand <= bott], np.linspace(0, 1, 9)))
        for t in ts:
            m = intra & (slack >= t)
            if not connects(m): continue
            worst = np.inf
            for b in range(nb):
                vb = np.where(lab == b)[0]
                if len(vb) < 2: continue
                eb = np.where(m & (lab[i] == b))[0]
                pos = {v: k for k, v in enumerate(vb)}
                Db = np.zeros((len(eb), len(vb)))
                Db[np.arange(len(eb)), [pos[v] for v in i[eb]]] = 1; Db[np.arange(len(eb)), [pos[v] for v in j[eb]]] = -1
                M = Db @ np.linalg.pinv(Db.T @ Db)
                rn = np.linalg.norm(M, axis=1)
                worst = min(worst, float(np.min(slack[eb] / (rn * 2 * kappa + 2 * lam * Lphi))))
            best = max(best, worst)
    rho_er = min(r_sep, best) if np.isfinite(bott) else r_sep
    return dict(sep=float(sep), r_sep=float(r_sep), bott=float(bott), kappa=float(kappa), nC=int(nC),
                rho_tree=float(rho_tree), rho_er=float(rho_er))


if __name__ == '__main__':
    did, outp = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(f'results/e3/c_{did}.jsonl')]
    X, y, name = load(did, 600)  # E3 used nmax=600
    assert len(X) == rows[0]['n']
    out = []
    for r0 in rows:
        lam, r = r0['lam'], r0['r']; f = fit(X, y, lam, r)
        v = rho_variants(f, lam, r, len(X)); v.update(did=int(did), dataset=name, n=len(X), lam=lam, n_balls=int(f['lab'].max() + 1),
                                                      rho_paper=r0['rho'], emp_break_min=r0['emp_break_min'])
        for dec in (6, 5, 4):
            dX = float(np.linalg.norm(np.round(X, dec) - X)); v[f'round{dec}_norm'] = dX
            v[f'round{dec}_cert_er'] = bool(dX < v['rho_er'])
            v[f'round{dec}_same'] = bool(ARI(f['lab'], fit(np.round(X, dec), y, lam, r)['lab']) == 1.0)
        if v['rho_er'] > 0:
            E = np.random.default_rng(99).standard_normal(X.shape); E *= 0.99 * v['rho_er'] / np.linalg.norm(E)
            v['cert_er_holds'] = bool(ARI(f['lab'], fit(X + E, y, lam, r, 1e-11)['lab']) == 1.0)
        print(json.dumps(v), flush=True); out.append(v)
    with open(outp, 'w') as fo:
        for v in out:
            fo.write(json.dumps(v) + '\n')
