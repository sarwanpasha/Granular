"""E18: a recent granular-ball generator, Xia et al., "An efficient and adaptive granular-ball generation method
in classification problem" (IEEE TNNLS 2024, arXiv 2201.04343), reimplemented from the paper (no code available).
 GB-acc : Algorithm 1 (acceleration). A ball whose purity is below 1 is split into k = #labels children: the
          ball's own division point plus one random point of every other label present; points go to the
          nearest division point; children keep their division point. Repeat until the ball count stops
          increasing, then a global division (every point to its nearest final division point).
 GB-adp : Algorithm 2 (adaptive), line 8 as printed: a split is accepted iff the weighted purity sum of the
          children exceeds the parent's purity, or the parent's purity is <= T0 (purity of the whole data set). After each pass,
          heterogeneous balls with ||c_i - c_j|| <= r_i + r_j (r = max distance to the division point) are
          de-overlapped by splitting the larger of the two. Global division at the end.
 Choices not fixed by the paper: the root's division point is the data mean; the k-1 new points are drawn
 uniformly from each non-majority label present; empty children are dropped; at most 60 passes.
Then CGB (radius graph) is matched to each generator's ball count and evaluated with the same protocol.
usage: python e18.py <openml_id> <out.jsonl> <nmax> [njobs]
"""
import sys, os, json
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from bench import uniq
from bench_r import gen_r, lam_r


def purity(y):
    return np.bincount(y).max() / len(y)


def split(idx, c0, X, y, rng):
    yy = y[idx]; labs, cnt = np.unique(yy, return_counts=True)
    if len(labs) < 2: return None
    maj = labs[np.argmax(cnt)]; pts = [c0]
    for l in labs:
        if l == maj: continue
        cand = idx[yy == l]; pts.append(X[cand[rng.integers(len(cand))]])
    P = np.array(pts); a = np.argmin(((X[idx][:, None, :] - P[None]) ** 2).sum(2), axis=1)
    ch = [(idx[a == t], P[t]) for t in range(len(P)) if np.any(a == t)]
    return ch if len(ch) > 1 else None


def gb_generate(X, y, seed=0, adaptive=False, max_pass=60):
    rng = np.random.default_rng(seed); T0 = purity(y)
    balls = [(np.arange(len(X)), X.mean(0))]
    for _ in range(max_pass):
        n0 = len(balls); new = []
        for idx, c in balls:
            if purity(y[idx]) >= 1.0: new.append((idx, c)); continue
            ch = split(idx, c, X, y, rng)
            if ch is None: new.append((idx, c)); continue
            if adaptive:
                W = sum(np.bincount(y[ci]).max() for ci, _ in ch) / len(idx)
                if not (W > purity(y[idx]) or purity(y[idx]) <= T0):
                    new.append((idx, c)); continue
            new.extend(ch)
        balls = new
        if adaptive and len(balls) > 1:
            C = np.array([c for _, c in balls]); R = np.array([np.sqrt(((X[i] - c) ** 2).sum(1)).max() for i, c in balls])
            L = np.array([np.bincount(y[i]).argmax() for i, _ in balls]); S = np.array([len(i) for i, _ in balls])
            D = np.sqrt(((C[:, None, :] - C[None]) ** 2).sum(2))
            ov = (D <= R[:, None] + R[None]) & (L[:, None] != L[None]); np.fill_diagonal(ov, False)
            todo = set()
            for a, b in zip(*np.where(np.triu(ov))):
                todo.add(a if S[a] >= S[b] else b)
            out = []
            for t, (idx, c) in enumerate(balls):
                ch = split(idx, c, X, y, rng) if t in todo else None
                out.extend(ch if ch is not None else [(idx, c)])
            balls = out
        if len(balls) == n0: break
    P = np.array([c for _, c in balls])
    lab = np.empty(len(X), int)
    for s in range(0, len(X), 2048):
        lab[s:s + 2048] = np.argmin(((X[s:s + 2048, None, :] - P[None]) ** 2).sum(2), axis=1)
    return np.unique(lab, return_inverse=True)[1]  # compact: division points left empty by the global division are dropped


_ORIG = bench.gen


def gen_e18(method, X, y, seed=0, lam=None):
    if method in ('GB-acc', 'GB-adp'):
        lab = gb_generate(X, y, seed, adaptive=(method == 'GB-adp'))
        nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
        C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, X); C /= np.maximum(cnt, 1)[:, None]
        return lab, C[lab]
    if method == 'CGB-r': return gen_r(method, X, y, seed, lam)
    return _ORIG(method, X, y, seed, lam)


def job(method, X, y, did, lam=None, matched=None):
    bench.gen = gen_e18
    r = bench.evaluate(method, X, y, lam, 11)
    r.update(method=method, lam=lam, matched_to=matched, did=int(did))
    return r


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    X, y, name = bench.load(did, nmax)
    G = ['GB-acc', 'GB-adp']
    base = Parallel(n_jobs=nj)(delayed(job)(m, X, y, did) for m in G)
    targets = [int(gb_generate(X, y, 0, adaptive=(m == 'GB-adp')).max() + 1) for m in G]
    lams = Parallel(n_jobs=nj)(delayed(lam_r)(X, y, t) for t in targets)
    cg = Parallel(n_jobs=nj)(delayed(job)('CGB-r', X, y, did, l, m) for l, m in zip(lams, G))
    with open(outp, 'w') as fo:
        for r in base + cg:
            r['dataset'] = name; fo.write(json.dumps(r) + '\n')
    print('done', did, name, len(X), targets, flush=True)
