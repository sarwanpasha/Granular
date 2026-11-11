"""E7 on CPU: granular balls on DINOv2 embeddings. stage 'feat' | 'gb' | 'stab' | 'fewshot'
usage: python e7_cpu.py <stage> [njobs]
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = f'{ROOT}/data/vision'; FEAT = f'{ROOT}/feats'; RES = f'{ROOT}/results/e7'
for p in (DATA, FEAT, RES):
    os.makedirs(p, exist_ok=True)
MODEL = 'dinov2_vits14'


def extract():
    import torch, torchvision as tv, torchvision.transforms as T
    torch.set_num_threads(os.cpu_count())
    torch.hub.set_dir(f'{ROOT}/hub')
    model = torch.hub.load('facebookresearch/dinov2', MODEL).eval()
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    sets = {'cifar10_train': lambda: tv.datasets.CIFAR10(DATA, True, tf, download=True),
            'cifar10_test': lambda: tv.datasets.CIFAR10(DATA, False, tf, download=True),
            'cifar100_train': lambda: tv.datasets.CIFAR100(DATA, True, tf, download=True),
            'cifar100_test': lambda: tv.datasets.CIFAR100(DATA, False, tf, download=True),
            'svhn_test': lambda: tv.datasets.SVHN(DATA, 'test', tf, download=True)}
    for nm, mk in sets.items():
        out = f'{FEAT}/{MODEL}_{nm}.npz'
        if os.path.exists(out): continue
        ds = mk(); dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)
        F, Y = [], []; t = time.time()
        with torch.inference_mode():
            for x, y in dl:
                F.append(model(x).numpy()); Y.append(np.asarray(y))
        np.savez(out, X=np.concatenate(F).astype(np.float32), y=np.concatenate(Y))
        print(nm, 'done', round(time.time() - t), flush=True)


def l2n(X):
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float64)


def load(nm):
    z = np.load(f'{FEAT}/{MODEL}_{nm}.npz'); return l2n(z['X']), z['y'].astype(int)


def strat(y, n, seed=0):
    rng = np.random.default_rng(seed); idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]; idx.append(rng.choice(ic, min(len(ic), int(round(n * len(ic) / len(y)))), replace=False))
    return np.sort(np.concatenate(idx))


def dmat(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None], 0))


def auroc(s_in, s_out):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.zeros(len(s_in)), np.ones(len(s_out))], np.r_[s_in, s_out]))


def gb_eval(C, R, yb, dens, Xte, yte, oods):
    out = {}
    d = dmat(Xte, C)
    out['acc_rad'] = float(np.mean(yb[(d - R).argmin(1)] == yte)); out['acc_den'] = float(np.mean(yb[(d - dens).argmin(1)] == yte))
    s_in = (d - R).min(1)
    for o, Xo in oods.items():
        out[f'auroc_{o}'] = auroc(s_in, (dmat(Xo, C) - R).min(1))
    return out


def run_gb(nj):
    from joblib import Parallel, delayed
    from bench import balls, uniq
    from e2c_ablate import fit_variant
    from baselines import gbgpp, gbclassic
    from sklearn.linear_model import LogisticRegression
    for ds, oo in (('cifar10', ('cifar100_test', 'svhn_test')), ('cifar100', ('cifar10_test', 'svhn_test'))):
        Xtr, ytr = load(f'{ds}_train'); Xte, yte = load(f'{ds}_test')
        idx = strat(ytr, 20000); Xtr, ytr = Xtr[idx], ytr[idx]
        oods = {o: load(o)[0][:5000] for o in oo if os.path.exists(f'{FEAT}/{MODEL}_{o}.npz')}
        W = open(f'{RES}/gb_{ds}.jsonl', 'a')
        def w(d):
            d.update(ds=ds, n_train=len(Xtr)); W.write(json.dumps(d) + '\n'); W.flush(); print(json.dumps(d), flush=True)
        t = time.time(); D = dmat(Xte, Xtr); nn = np.argpartition(D, 50, axis=1)[:, :50]
        nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(D, nn, 1), 1), 1)
        votes = np.zeros((len(Xte), ytr.max() + 1))
        for k in range(20): np.add.at(votes, (np.arange(len(Xte)), ytr[nn[:, k]]), 1)
        rr = dict(method='kNN20', acc=float(np.mean(votes.argmax(1) == yte)), sec=time.time() - t)
        s_in = D[np.arange(len(Xte)), nn[:, -1]]
        for o, Xo in oods.items():
            Do = dmat(Xo, Xtr); rr[f'auroc_{o}'] = auroc(s_in, np.partition(Do, 49, axis=1)[:, 49])
        w(rr)
        cm = np.stack([Xtr[ytr == c].mean(0) for c in range(ytr.max() + 1)])
        rr = dict(method='NCM', acc=float(np.mean(dmat(Xte, cm).argmin(1) == yte)), n_balls=len(cm))
        s_in = dmat(Xte, cm).min(1)
        for o, Xo in oods.items(): rr[f'auroc_{o}'] = auroc(s_in, dmat(Xo, cm).min(1))
        w(rr)
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        w(dict(method='linear-probe', acc=float(lr.score(Xte, yte))))
        def one(meth, lam):
            t = time.time()
            if meth == 'GBG++': mem = uniq(gbgpp(Xtr, ytr))
            elif meth == 'GBclassic': mem = uniq(gbclassic(Xtr, ytr, seed=0))
            else: mem = uniq(fit_variant(Xtr, ytr, lam, meth)[0])
            sec = time.time() - t
            C, R, yb, dens = balls(mem, Xtr, ytr)
            r = dict(method=('CGB-' + meth) if lam else meth, lam=lam, n_balls=len(C), sec=sec)
            r.update(gb_eval(C, R, yb, dens, Xte, yte, oods)); return r
        jobs = [('GBG++', None), ('GBclassic', None)] + [(v, l) for v in ('knn10', 'radius') for l in (0.03, 0.1, 0.3, 1.0)]
        for r in Parallel(n_jobs=nj)(delayed(one)(m, l) for m, l in jobs):
            w(r)


def run_stab(nj, n=5000):
    from joblib import Parallel, delayed
    from bench import uniq
    from e2c_ablate import fit_variant
    from baselines import gbgpp, gbclassic
    from sklearn.metrics import adjusted_rand_score as ARI
    X, y = load('cifar10_train'); idx = strat(y, n, 1); X, y = X[idx], y[idx]
    W = open(f'{RES}/stab_cifar10.jsonl', 'a')
    def match(v, target):
        a, b = np.log(1e-3), np.log(300.0)
        for _ in range(14):
            mid = 0.5 * (a + b); nb = int(fit_variant(X, y, float(np.exp(mid)), v)[0].max() + 1)
            if nb > target: a = mid
            else: b = mid
        return float(np.exp(0.5 * (a + b)))
    def stab(f, ref, seed):
        rng = np.random.default_rng(seed); o = {}
        for eps in (0.002, 0.01):
            o[f'jit{eps}'] = float(np.mean([1 - ARI(ref, f(X + eps * rng.standard_normal(X.shape) / np.sqrt(X.shape[1]), y)) for _ in range(3)]))
        v = []
        for _ in range(3):
            keep = np.sort(rng.choice(n, int(0.95 * n), replace=False)); v.append(1 - ARI(ref[keep], f(X[keep], y[keep])))
        o['sub95'] = float(np.mean(v)); return o
    BASES = os.environ.get('STAB_BASES', 'GBG++,GBclassic').split(',')
    VARS = tuple(os.environ.get('STAB_VARS', 'knn10,radius').split(','))
    for base, fb in [x for x in (('GBG++', lambda A, b: uniq(gbgpp(A, b))), ('GBclassic', lambda A, b: uniq(gbclassic(A, b, seed=0)))) if x[0] in BASES]:
        ref = fb(X, y); nb = ref.max() + 1
        lams = Parallel(n_jobs=2)(delayed(match)(v, nb) for v in VARS)
        cfg = [(base, fb, None)] + [(v, (lambda A, b, v=v, l=l: uniq(fit_variant(A, b, l, v)[0])), l) for v, l in zip(VARS, lams)]
        res = Parallel(n_jobs=3)(delayed(lambda nm, f, l: dict(method=nm, lam=l, matched_to=base, n_balls=int(f(X, y).max() + 1), **stab(f, f(X, y), 0)))(nm, f, l) for nm, f, l in cfg)
        for r in res:
            W.write(json.dumps(r) + '\n'); W.flush(); print(json.dumps(r), flush=True)


def run_fewshot(nj, shots=(1, 5, 10), reps=5):
    from bench import balls, uniq
    from e2c_ablate import fit_variant
    from baselines import gbgpp
    W = open(f'{RES}/fewshot2.jsonl', 'a')
    for ds in ('cifar10', 'cifar100'):
        Xtr, ytr = load(f'{ds}_train'); Xte, yte = load(f'{ds}_test')
        for s in shots:
            for rep in range(reps):
                rng = np.random.default_rng(rep)
                idx = np.concatenate([rng.choice(np.where(ytr == c)[0], s, replace=False) for c in np.unique(ytr)])
                X, y = Xtr[idx], ytr[idx]
                cm = np.stack([X[y == c].mean(0) for c in np.unique(y)])
                r = dict(ds=ds, shots=s, rep=rep, NCM=float(np.mean(dmat(Xte, cm).argmin(1) == yte)),
                         NN1=float(np.mean(y[dmat(Xte, X).argmin(1)] == yte)))
                if s > 1:
                    for lam in (0.1, 0.3, 1.0, 3.0):
                        mem = uniq(fit_variant(X, y, lam, 'radius')[0]) if len(X) > 12 else np.arange(len(X))
                        C, R, yb, dens = balls(mem, X, y)
                        d = dmat(Xte, C); r[f'CGB{lam}_balls'] = len(C)
                        r[f'CGB{lam}_rad'] = float(np.mean(yb[(d - R).argmin(1)] == yte)); r[f'CGB{lam}_den'] = float(np.mean(yb[(d - dens).argmin(1)] == yte))
                        r[f'CGB{lam}_ctr'] = float(np.mean(yb[d.argmin(1)] == yte)); r[f'CGB{lam}'] = max(r[f'CGB{lam}_rad'], r[f'CGB{lam}_den'])
                    mem = uniq(gbgpp(X, y)); C, R, yb, dens = balls(mem, X, y); d = dmat(Xte, C)
                    r['GBG++'] = float(max(np.mean(yb[(d - R).argmin(1)] == yte), np.mean(yb[(d - dens).argmin(1)] == yte)))
                W.write(json.dumps(r) + '\n'); W.flush(); print(json.dumps(r), flush=True)


if __name__ == '__main__':
    st = sys.argv[1]; nj = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    if st == 'feat': extract()
    elif st == 'gb': run_gb(nj)
    elif st == 'stab': run_stab(nj)
    elif st == 'fewshot': run_fewshot(nj)
