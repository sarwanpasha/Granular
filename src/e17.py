"""E17: classical data-summarization baselines at the GBG++ ball count, benchmark protocol (bench.evaluate:
order, seed, jitter 0.01/0.05, sub95, drift, 5-fold CV accuracy).
 KM-class     : k-means per class (k-means++, n_init=10, seed passed through), k allocated to classes by size.
 Ward-class   : Ward agglomerative clustering per class (deterministic, no seed).
 KCenter-class: farthest-first traversal per class, started at the point nearest the class mean (deterministic).
Per-class k: largest-remainder allocation of the target proportional to class size, at least 1, at most n_c.
The 'lam' argument of bench.gen carries the target ball count.
usage: python e17.py <openml_id> <out.jsonl> <nmax> [njobs]
"""
import sys, os, json
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, AgglomerativeClustering
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from bench import uniq


def alloc(y, target):
    cls, cnt = np.unique(y, return_counts=True)
    target = int(min(max(target, len(cls)), len(y)))
    q = cnt / cnt.sum() * target; k = np.maximum(np.floor(q).astype(int), 1); k = np.minimum(k, cnt)
    rem = q - np.floor(q)
    for i in np.argsort(-rem, kind='stable'):
        if k.sum() >= target: break
        if k[i] < cnt[i]: k[i] += 1
    while k.sum() < target:
        i = int(np.argmax(cnt - k)); k[i] += 1
    return cls, k


def per_class(X, y, target, fit):
    lab = np.empty(len(X), int); off = 0
    cls, ks = alloc(y, target)
    for c, k in zip(cls, ks):
        idx = np.where(y == c)[0]
        l = np.zeros(len(idx), int) if k <= 1 else (np.arange(len(idx)) if k >= len(idx) else fit(X[idx], int(k)))
        l = uniq(l); lab[idx] = l + off; off += l.max() + 1
    return lab


def km(seed):
    return lambda Xc, k: KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed).fit(Xc).labels_


def ward(Xc, k):
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit(Xc).labels_


def kcenter(Xc, k):
    start = int(np.argmin(((Xc - Xc.mean(0)) ** 2).sum(1)))
    d = ((Xc - Xc[start]) ** 2).sum(1); centers = [start]
    for _ in range(k - 1):
        nxt = int(np.argmax(d)); centers.append(nxt); d = np.minimum(d, ((Xc - Xc[nxt]) ** 2).sum(1))
    C = Xc[centers]
    return np.argmin(((Xc[:, None, :] - C[None]) ** 2).sum(2), axis=1)


_ORIG = bench.gen


def gen_e17(method, X, y, seed=0, lam=None):
    if method == 'KM-class': lab = per_class(X, y, lam, km(seed))
    elif method == 'Ward-class': lab = per_class(X, y, lam, ward)
    elif method == 'KCenter-class': lab = per_class(X, y, lam, kcenter)
    else: return _ORIG(method, X, y, seed, lam)
    lab = uniq(lab); nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, X); C /= np.maximum(cnt, 1)[:, None]
    return lab, C[lab]


def job(method, X, y, did, target):
    bench.gen = gen_e17
    r = bench.evaluate(method, X, y, target, 11)
    r.update(method=method, target=int(target), did=int(did))
    return r


if __name__ == '__main__':
    did, outp, nmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nj = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    X, y, name = bench.load(did, nmax)
    target = [json.loads(l) for l in open(f'results/bench/b_{did}.jsonl') if json.loads(l)['method'] == 'GBG++'][0]['n_balls']
    meths = ['KM-class', 'Ward-class', 'KCenter-class']
    res = Parallel(n_jobs=nj)(delayed(job)(m, X, y, did, target) for m in meths)
    with open(outp, 'w') as fo:
        for r in res:
            r['dataset'] = name; fo.write(json.dumps(r) + '\n')
    print('done', did, name, len(X), target, flush=True)
