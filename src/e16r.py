"""E16r: the RGGB-s cells of E16 that were not obtained on the GPU workstation (timeout > 1 h or skipped / stopped for memory),
run on CPU nodes with up to 256 GB RAM and no per-run time limit.
Same data, split and nested training prefixes as e16w.py (bench.load == e9g.load; seeds identical), same
accuracy rules. A GBG++ fit at n=12500 is recorded as a data check against the e16w.py rows.
usage: python e16r.py <openml_id> <n> <with_stab 0|1> <out.jsonl>
"""
import sys, os, json, time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from baselines import gbgpp, rggb_s, _split_pure, _to_mem


def rggb_s_chunked(X, y, seed=0, cw=0.5, iters=5, chunk=4096):
    """baselines.rggb_s with the n x (#balls) score matrix computed in row blocks (same formula, same argmax);
    the original allocates the full matrix, which needs 361 GiB at n=400000 on Covertype."""
    rng = np.random.default_rng(seed); n = len(X)
    bl = _split_pure(X, y, rng, 'rand')
    for _ in range(iters):
        C = np.array([X[b].mean(0) for b in bl])
        R = np.array([np.sqrt(((X[b] - X[b].mean(0)) ** 2).sum(1)).mean() if len(b) > 1 else 0 for b in bl]) + 1e-9
        lb = np.array([np.bincount(y[b]).argmax() for b in bl]); c2 = (C ** 2).sum(1)
        a = np.empty(n, int)
        for s in range(0, n, chunk):
            Z = X[s:s + chunk]
            d = np.maximum((Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + c2[None], 0)
            sc = cw / (1 + d) - (1 - cw) * np.sqrt(d) / R[None]
            sc[lb[None, :] != y[s:s + chunk, None]] = -np.inf
            a[s:s + chunk] = sc.argmax(1)
        bl = [np.where(a == j)[0] for j in np.unique(a)]
    return _to_mem(n, bl)


def compact(lab):
    return np.unique(np.asarray(lab), return_inverse=True)[1]


def acc(lab, X, y, Xt, yt):
    nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, X); C /= cnt[:, None]
    R = np.zeros(nb); np.add.at(R, lab, np.linalg.norm(X - C[lab], axis=1)); R /= cnt
    yb = np.array([np.bincount(y[lab == b]).argmax() for b in range(nb)])
    out = {}
    for rule in ('rad', 'den'):
        sub = R if rule == 'rad' else cnt / cnt.sum(); pr = np.empty(len(Xt), int); c2 = (C ** 2).sum(1)
        for s in range(0, len(Xt), 1024):
            Z = Xt[s:s + 1024]
            d = np.sqrt(np.maximum((Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + c2[None], 0)) - sub[None]
            pr[s:s + 1024] = yb[np.argmin(d, 1)]
        out[rule] = float(np.mean(pr == yt))
    return out


if __name__ == '__main__':
    did, n, stab, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    X, y, name = bench.load(did, 10 ** 9)
    rng = np.random.default_rng(0); te = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]; te.append(rng.choice(ic, max(1, int(round(min(0.2, 20000 / len(y)) * len(ic)))), replace=False))
    te = np.sort(np.concatenate(te)); tr = np.setdiff1d(np.arange(len(y)), te)
    Xt, yt = X[te], y[te]; perm = tr[rng.permutation(len(tr))]
    fo = open(outp, 'a')
    def emit(d):
        d.update(did=int(did), dataset=name); fo.write(json.dumps(d) + '\n'); fo.flush(); print(json.dumps(d), flush=True)
    Xc, yc = X[perm[:12500]], y[perm[:12500]]; lab = compact(gbgpp(Xc, yc))
    emit(dict(kind='check', method='GBG++', n=12500, n_train_total=len(perm), n_balls=int(lab.max() + 1), **{'acc_' + k: v for k, v in acc(lab, Xc, yc, Xt, yt).items()}))
    assert n <= len(perm)
    R_ = rggb_s
    if os.environ.get('RGGB_CHUNKED') == '1':
        Xv, yv = X[perm[:50000]], y[perm[:50000]]
        a0 = compact(rggb_s(Xv, yv, seed=0)); a1 = compact(rggb_s_chunked(Xv, yv, seed=0))
        emit(dict(kind='check', method='RGGB-s chunked vs original', n=50000, identical=bool(np.array_equal(a0, a1)), ari=float(ARI(a0, a1))))
        R_ = rggb_s_chunked
    Xn, yn = X[perm[:n]], y[perm[:n]]
    t = time.time(); lab = compact(R_(Xn, yn, seed=0)); sec = time.time() - t
    d = dict(kind='fit', method='RGGB-s', impl='chunked' if R_ is not rggb_s else 'original', n=n, sec=sec, timeout=False, n_balls=int(lab.max() + 1), **{'acc_' + k: v for k, v in acc(lab, Xn, yn, Xt, yt).items()})
    if stab:
        k = np.sort(np.random.default_rng(1).choice(n, int(0.95 * n), replace=False))
        d['sub95'] = float(1 - ARI(lab[k], compact(R_(Xn[k], yn[k], seed=0))))
        p = np.random.default_rng(2).permutation(n); m3 = np.empty(n, int); m3[p] = compact(R_(Xn[p], yn[p], seed=0))
        d['order'] = float(1 - ARI(lab, m3))
    emit(d)
