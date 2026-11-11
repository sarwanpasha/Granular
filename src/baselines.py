"""Baseline GB generators returning point->ball membership (partition; -1 = uncovered).
gbgpp: faithful re-transcription of authors' GBG_plusplus.py (CherylTse/GBGplusplus).
gbclassic: Xia et al. 2019 k-division split-until-pure (randomized k-means init).
"""
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans


def _labpur(data):
    if len(data) > 1:
        c = Counter(data[:, 0]); lab = max(c, key=c.get); return lab, c[lab] / len(data)
    return data[0][0], 1.0


def _cr(data):
    c = np.mean(data, axis=0); return c, np.mean(np.sqrt(np.sum((data - c) ** 2, axis=1)))


class _Ball:
    def __init__(self, data):
        self.data = data
        self.center, self.radius = _cr(data[:, 1:])
        self.label, self.purity = _labpur(data)


def _omega(data):
    c = Counter(data[:, 0]); label = max(c, key=c.get)
    og = data[data[:, 0] == label]
    center = np.mean(og[:, 1:], axis=0)
    dis = np.around(np.sqrt(np.sum((data[:, 1:] - center) ** 2, axis=1)), 6)
    radius = np.around(np.mean(dis[data[:, 0] == label]), 6)
    inn = dis <= radius
    return data[inn], data[~inn]


def _remove_conflicts(bl):
    rm = []
    for i in range(len(bl) - 1):
        if any(bl[i] is b for b in rm):
            continue
        for j in range(i + 1, len(bl)):
            if any(bl[j] is b for b in rm):
                continue
            if bl[i].label != bl[j].label and np.sqrt(np.sum((bl[i].center - bl[j].center) ** 2)) <= abs(bl[i].radius - bl[j].radius):
                bl[j].data = np.concatenate((bl[j].data, bl[i].data))
                bl[j].label, bl[j].purity = _labpur(bl[j].data)
                bl[j].center, bl[j].radius = _cr(bl[j].data[:, 1:])
                rm.append(bl[i]); break
    return [b for b in bl if not any(b is r for r in rm)]


def _split(gb):
    todo = gb; ln = len(np.unique(gb[:, 0])); tn = len(todo); bl = []; out = []
    while ln != tn:
        cl, tmp = _omega(todo)
        if len(cl) > 1:
            bl.append(_Ball(cl))
        todo = tmp; tn = len(todo)
        if tn > 1:
            ln = len(np.unique(todo[:, 0]))
        else:
            break
    if len(bl) > 1:
        out = _remove_conflicts(bl)
    return out


def gbgpp(X, y, purity=1.0):
    """Returns membership array (n,), -1 for points dropped by GBG++."""
    n = len(X)
    feats = np.column_stack([y.astype(float), X])
    GB = [_Ball(feats)]; i = 0; num = 1
    while True:
        if GB[i].purity < purity:
            ns = _split(GB[i].data)
            if len(ns) > 1:
                GB[i] = ns[0]; GB.extend(ns[1:])
            elif len(ns) == 1 and len(ns[0].data) != len(GB[i].data):
                GB.pop(i); GB.append(ns[0])
            else:
                GB.pop(i)
            num = len(GB)
        else:
            i += 1
        if i >= num:
            break
    pos = defaultdict(list)
    for idx, row in enumerate(feats):
        pos[row.tobytes()].append(idx)
    mem = -np.ones(n, dtype=int)
    for b, ball in enumerate(GB):
        for row in ball.data:
            lst = pos[row.tobytes()]
            if lst:
                mem[lst.pop()] = b
    return mem


def gbclassic(X, y, purity=1.0, seed=0, min_size=1):
    rng = np.random.RandomState(seed)
    K = len(np.unique(y))
    stack = [np.arange(len(X))]; balls = []
    while stack:
        idx = stack.pop()
        c = Counter(y[idx]); p = max(c.values()) / len(idx)
        if p >= purity or len(idx) <= max(min_size, K):
            balls.append(idx); continue
        km = KMeans(n_clusters=min(K, len(idx)), n_init=1, init='random',
                    random_state=rng.randint(1 << 30)).fit(X[idx])
        parts = [idx[km.labels_ == t] for t in range(km.n_clusters)]
        parts = [q for q in parts if len(q)]
        if len(parts) <= 1:
            balls.append(idx); continue
        stack.extend(parts)
    mem = -np.ones(len(X), dtype=int)
    for b, idx in enumerate(balls):
        mem[idx] = b
    return mem


# ---------------- RGGB-style / ScOrGBG-style (our re-implementations from the audit code) ----------------
def _split_pure(X, y, rng, init):
    n = len(X); stack = [np.arange(n)]; out = []
    while stack:
        idx = stack.pop(); vals, cnts = np.unique(y[idx], return_counts=True)
        if len(vals) <= 1 or len(idx) <= 1:
            out.append(idx); continue
        k = len(vals); Xs = X[idx]
        if init == 'pp':
            c = [Xs[rng.integers(len(idx))]]; d2 = ((Xs - c[0]) ** 2).sum(1)
            for _ in range(1, k):
                p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(idx), 1 / len(idx))
                c.append(Xs[rng.choice(len(idx), p=p)]); d2 = np.minimum(d2, ((Xs - c[-1]) ** 2).sum(1))
            C = np.stack(c)
        else:
            C = Xs[rng.choice(len(idx), size=k, replace=False)]
        a = ((Xs[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        subs = [idx[a == j] for j in np.unique(a)]
        if len(subs) <= 1:
            out.append(idx); continue
        stack.extend(subs)
    return out


def _to_mem(n, bl):
    m = -np.ones(n, dtype=int)
    for b, idx in enumerate(bl):
        m[idx] = b
    return m


def scorgbg_s(X, y, seed=0):
    """k-means++-seeded split-to-purity (membership-determining stage of ScOrGBG)."""
    return _to_mem(len(X), _split_pure(X, y, np.random.default_rng(seed), 'pp'))


def rggb_s(X, y, seed=0, cw=0.5, iters=5):
    """random-init split + justifiable-granularity style regeneration (RGGB-style)."""
    rng = np.random.default_rng(seed); n = len(X)
    bl = _split_pure(X, y, rng, 'rand')
    for _ in range(iters):
        C = np.array([X[b].mean(0) for b in bl])
        R = np.array([np.sqrt(((X[b] - X[b].mean(0)) ** 2).sum(1)).mean() if len(b) > 1 else 0 for b in bl]) + 1e-9
        lb = np.array([np.bincount(y[b]).argmax() for b in bl])
        d = np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None], 0)
        sc = cw / (1 + d) - (1 - cw) * np.sqrt(d) / R[None]
        sc[lb[None, :] != y[:, None]] = -np.inf
        a = sc.argmax(1)
        bl = [np.where(a == j)[0] for j in np.unique(a)]
    return _to_mem(n, bl)
