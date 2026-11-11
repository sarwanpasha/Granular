import os, sys, json, time, hashlib, numpy as np, torch, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
dv = 'cuda'; f8 = torch.float64; EPS = (0.01, 0.05); R = 5
TG = {554: [('GBG++', 614), ('GBclassic', 649), ('RGGB-s', 704), ('ScOrGBG-s', 967)],
      1478: [('GBG++', 617), ('GBclassic', 531), ('RGGB-s', 616), ('ScOrGBG-s', 793)]}


def load(did, nmax, seed=0):
    import pandas as pd
    d = fetch_openml(data_id=int(did), as_frame=True, parser='auto', data_home='data')
    df = d.data.copy()
    for c in df.columns:
        if str(df[c].dtype) in ('category', 'object', 'bool'):
            df[c] = df[c].astype('category')
    num = df.select_dtypes(include='number'); cat = df.select_dtypes(exclude='number')
    num = num.fillna(num.median()); parts = [num]
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
        rng = np.random.default_rng(seed); idx = []
        for c in np.unique(y):
            ic = np.where(y == c)[0]; k = max(1, int(round(nmax * len(ic) / len(y))))
            idx.append(rng.choice(ic, min(k, len(ic)), replace=False))
        idx = np.sort(np.concatenate(idx)); X, y = X[idx], y[idx]
    return X, y, d.details.get('name', str(did))


def radius_graph(X, y, r):
    A = NearestNeighbors(radius=r).fit(X).radius_neighbors_graph(X, mode='distance')
    A = sp.triu(A, k=1).tocoo(); i, j, d = A.row, A.col, A.data
    t = np.clip(1 - (d / r) ** 2, 0, None); w = t * t
    m = (w > 0) & (y[i] == y[j]); o = np.lexsort((j[m], i[m]))
    return i[m][o], j[m][o], w[m][o]


def solve(X, i, j, gam, tol=1e-8, max_iter=20000):
    n, d = X.shape; m = len(i)
    I = torch.tensor(i, device=dv, dtype=torch.long); J = torch.tensor(j, device=dv, dtype=torch.long)
    Xt = torch.tensor(X, device=dv, dtype=f8)
    Dm = lambda U: U[I] - U[J]
    ar = torch.arange(m, device=dv); DTs = torch.sparse_coo_tensor(torch.stack([torch.cat([I, J]), torch.cat([ar, ar])]), torch.cat([torch.ones(m, device=dv, dtype=f8), -torch.ones(m, device=dv, dtype=f8)]), (n, m)).coalesce().to_sparse_csr()
    def DT(L):
        return DTs @ L
    if m > 0:
        v = np.random.default_rng(0).standard_normal(n); v /= np.linalg.norm(v); v = torch.tensor(v, device=dv, dtype=f8)[:, None]
        for _ in range(60):
            u = DT(Dm(v)); L = u.norm(); v = u / (L + 1e-300)
        L = 1.02 * float(L)
    else:
        L = 1.0
    nu = 1.0 / L; DX = Dm(Xt)
    Lam = torch.zeros((m, d), device=dv, dtype=f8); Y = Lam.clone(); t = 1.0
    g = torch.tensor(gam, device=dv, dtype=f8)[:, None]; gt = g[:, 0]
    proj = lambda Z: Z * torch.clamp(g / torch.clamp(Z.norm(dim=1, keepdim=True), min=1e-300), max=1.0)
    gap = float('inf'); it = 0
    for it in range(1, max_iter + 1):
        U = Xt - DT(Y); Lnew = proj(Y + nu * Dm(U))
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); Y = Lnew + ((t - 1) / tn) * (Lnew - Lam); Lam, t = Lnew, tn
        if it % 50 == 0 or it == max_iter:
            U = Xt - DT(Lam); DU = Dm(U)
            P = 0.5 * ((U - Xt) ** 2).sum() + (gt * DU.norm(dim=1)).sum()
            Gd = (Lam * DX).sum() - 0.5 * (DT(Lam) ** 2).sum(); gap = float(P - Gd)
            if gap <= tol * max(1.0, abs(float(P))): break
            if float(((Lnew - Y) * (Lam - Lnew)).sum()) > 0: Y = Lam.clone(); t = 1.0
    U = Xt - DT(Lam)
    Z = Lam + Dm(U) / L; fused = (Z.norm(dim=1) <= gt).cpu().numpy()
    return U.cpu().numpy(), fused, it


GC = {}; ST = {'fits': 0, 'iters': 0, 'maxit': 0, 'hits': 0}; os.makedirs('cache', exist_ok=True)


def fitr(X, y, lam):
    key = hashlib.md5(X.tobytes() + y.astype(np.int64).tobytes()).hexdigest()
    if key not in GC:
        if len(GC) > 4: GC.clear()
        r = np.median(NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1])
        GC[key] = radius_graph(X, y, r)
    i, j, w = GC[key]
    U, fused, it = solve(X, i, j, lam * w)
    ST['fits'] += 1; ST['iters'] += it; ST['maxit'] += int(it == 20000)
    A = sp.csr_matrix((np.ones(fused.sum()), (i[fused], j[fused])), shape=(len(X), len(X)))
    return connected_components(A, directed=False)[1], U


def gen(X, y, lam):
    fp = os.path.join('cache', hashlib.md5(X.tobytes() + y.astype(np.int64).tobytes() + repr(float(lam)).encode()).hexdigest() + '.npz')
    if os.path.exists(fp):
        z = np.load(fp); ST['hits'] += 1; return z['lab'], z['C'][z['lab']]
    lk = fp + '.lock'
    try:
        os.close(os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); mine = True
    except FileExistsError:
        mine = False
    if not mine:
        tw = time.time()
        while not os.path.exists(fp) and time.time() - tw < 7200: time.sleep(5)
        if os.path.exists(fp):
            z = np.load(fp); ST['hits'] += 1; return z['lab'], z['C'][z['lab']]
    lab, U = fitr(X, y, lam); nb = lab.max() + 1; cnt = np.bincount(lab, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, lab, U); C /= np.maximum(cnt, 1)[:, None]
    np.savez(fp + '.tmp.npz', lab=lab, C=C); os.replace(fp + '.tmp.npz', fp)
    if os.path.exists(lk): os.remove(lk)
    return lab, C[lab]


def balls(mem, X, y):
    nb = mem.max() + 1; cnt = np.bincount(mem, minlength=nb)
    C = np.zeros((nb, X.shape[1])); np.add.at(C, mem, X); C /= np.maximum(cnt, 1)[:, None]
    Rr = np.zeros(nb); np.add.at(Rr, mem, np.linalg.norm(X - C[mem], axis=1)); Rr /= np.maximum(cnt, 1)
    yb = np.array([np.bincount(y[mem == b]).argmax() for b in range(nb)])
    return C, Rr, yb, cnt / cnt.sum()


def predict(C, Rr, yb, dens, Xt, rule):
    out = np.empty(len(Xt), dtype=yb.dtype)
    for s in range(0, len(Xt), 1024):
        Z = Xt[s:s + 1024]
        d = np.sqrt(np.maximum((Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None], 0))
        d = d - (Rr[None] if rule == 'rad' else dens[None]); out[s:s + 1024] = yb[np.argmin(d, 1)]
    return out


def evaluate(X, y, lam, seed0):
    rng = np.random.default_rng(seed0); t0 = time.time(); n = len(X); out = {}
    ref, Zref = gen(X, y, lam); out['n_balls'] = int(ref.max() + 1); out['gen_sec'] = time.time() - t0
    v = []
    for r in range(R):
        p = rng.permutation(n); m = np.empty(n, int); m[p] = gen(X[p], y[p], lam)[0]; v.append(1 - ARI(ref, m))
    out['order'] = float(np.mean(v))
    for eps in EPS:
        v = []; dr = []
        for r in range(R):
            E = eps * rng.standard_normal(X.shape); mp, Zp = gen(X + E, y, lam)
            v.append(1 - ARI(ref, mp)); dr.append(np.linalg.norm(Zp - Zref) / np.linalg.norm(E))
        out[f'jit{eps}'] = float(np.mean(v)); out[f'drift{eps}'] = float(np.mean(dr))
    v = []
    for r in range(R):
        keep = np.sort(rng.choice(n, int(0.95 * n), replace=False)); v.append(1 - ARI(ref[keep], gen(X[keep], y[keep], lam)[0]))
    out['sub95'] = float(np.mean(v))
    skf = StratifiedKFold(5, shuffle=True, random_state=0); acc = {'rad': [], 'den': []}; nbs = []
    for tr, te in skf.split(X, y):
        mem = gen(X[tr], y[tr], lam)[0]; C, Rr, yb, dens = balls(mem, X[tr], y[tr]); nbs.append(len(C))
        for rule in acc: acc[rule].append(np.mean(predict(C, Rr, yb, dens, X[te], rule) == y[te]))
    out['acc_rad'] = float(np.mean(acc['rad'])); out['acc_den'] = float(np.mean(acc['den']))
    out['acc'] = max(out['acc_rad'], out['acc_den']); out['cv_balls'] = float(np.mean(nbs)); out['sec'] = time.time() - t0
    return out


def lam_r(X, y, target):
    a, b = np.log(1e-3), np.log(300.0); best = None
    for _ in range(14):
        mid = 0.5 * (a + b); lam = float(np.exp(mid)); nb = int(gen(X, y, lam)[0].max() + 1)
        print('bis', lam, nb, flush=True)
        if best is None or abs(nb - target) < abs(best[1] - target): best = (lam, nb)
        if nb == target: break
        if nb > target: a = mid
        else: b = mid
    return best[0]


if __name__ == '__main__':
    did = int(sys.argv[1]); X, y, name = load(did, 2000)
    if sys.argv[2] == 'ck':
        f = np.zeros(len(y), int)
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)): f[te] = k
        print('CK', did, name, X.shape, X.dtype, hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()[:12],
              hashlib.md5(y.astype(np.int64).tobytes()).hexdigest()[:12], hashlib.md5(f.astype(np.int64).tobytes()).hexdigest()[:12], repr(float(X.sum())))
        sys.exit()
    k = int(sys.argv[2]); b, tg = TG[did][k]; t0 = time.time()
    if os.path.exists('e9g_results.jsonl') and any(json.loads(l)['did'] == did and json.loads(l)['matched_to'] == b for l in open('e9g_results.jsonl') if l.strip()): print('already done', did, b); sys.exit()
    lam = lam_r(X, y, tg); r = evaluate(X, y, lam, 7 + k)
    r.update(did=did, dataset=name, n=len(X), d=X.shape[1], method='CGB-r', lam=lam, matched_to=b, target=tg,
             total_sec=time.time() - t0, fits=ST['fits'], mean_iters=ST['iters'] / max(ST['fits'], 1), fits_at_maxit=ST['maxit'], cache_hits=ST['hits'],
             device=torch.cuda.get_device_name(), solver='torch-float64')
    print(json.dumps(r), flush=True)
    with open('e9g_results.jsonl', 'a') as fo: fo.write(json.dumps(r) + '\n')
