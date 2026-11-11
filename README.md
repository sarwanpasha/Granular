# Granular

Convex Granular Balls (CGB): stable granular-ball generation by convex fusion.

## What this code does

Granular-ball (GB) computing replaces the samples of a labeled data set by a small number of pure balls (center, radius, label) and trains learners on the balls. Existing generators split balls greedily, so the balls change when the rows are reordered, a seed changes or a few points are removed.

CGB obtains the balls from the convex problem

```
min_U  1/2 ||U - X||_F^2 + lambda * sum_{(i,j) in E} w_ij ||u_i - u_j||
```

where `X` holds the samples as rows, `U` holds one representative per sample, `E` contains the pairs of samples with the same label whose distance is below a radius `r` (the median distance to the 10th nearest neighbor), and `w_ij = (1 - d_ij^2 / r^2)^2` is a smooth weight. The objective is strongly convex, so the solution is unique and does not depend on row order or seeds. The problem is solved in the dual with FISTA; an edge is fused when its dual optimality condition says that its two representatives coincide, and the balls are the connected components of the fused edges. Every ball is pure because `E` has no edge between different labels. `lambda` sets the granularity: `lambda = 0` gives one ball per sample, and `lambda = inf` gives the connected components of `E`.

## Requirements

- Python 3.9 or later
- The packages in `requirements.txt`:

```
pip install -r requirements.txt
```

- A CUDA GPU and a CUDA build of PyTorch only for the GPU scripts (`e9g.py`, `e16w.py`, `e21.py`, `e21b.py`). Everything else runs on CPU.

All commands below are run from the repository root. Data sets are downloaded from OpenML on first use and cached in `./data`. Results are written to `./results/<experiment>/` as JSON Lines files (one JSON object per line). Create the output folder before running a script, for example `mkdir -p results/bench`.

## Quick start: granulate one data set

```python
import sys; sys.path.insert(0, 'src')
import numpy as np
import bench
from bench_r import gen_r

X, y, name = bench.load(11, 3000)          # OpenML 11 (balance-scale), min-max scaled, de-duplicated
labels, Z = gen_r('CGB-r', X, y, 0, 1.0)   # radius-graph CGB with lambda = 1.0
print(name, X.shape, 'balls:', len(np.unique(labels)))
```

`labels[i]` is the ball of sample `i`, and row `i` of `Z` is the center of that ball.

## Reproducing the experiments

Each experiment script takes an OpenML data set ID and writes one JSON Lines file per data set. An aggregation script then reads all files of the experiment and prints the summary table (medians over data sets, wins/ties/losses of CGB, two-sided Wilcoxon signed-rank p-values and Holm-adjusted p-values).

The 56 benchmark data sets (OpenML IDs) are:

```
11 12 14 15 16 18 22 23 28 29 31 32 37 44 46 54 151 182 188 300 307 458 554 1049 1050 1053 1063
1067 1068 1461 1462 1464 1468 1478 1480 1485 1487 1489 1494 1497 1501 1510 1590 4534 4538 6332
23517 40499 40670 40701 40966 40975 40982 40983 40984 40994
```

Data sets with more than 3,000 rows are subsampled with stratification to 3,000 rows (2,000 for the ten largest or highest-dimensional ones); this is done inside `bench.load`, so `nmax` is always 3000 in the commands below. The optional last argument `njobs` is the number of parallel processes.

### 1. Main benchmark at matched granularity

```
mkdir -p results/bench results/bench_r
for ID in 11 12 14 ...; do
  python src/bench.py   $ID results/bench/b_$ID.jsonl   3000 4    # GBG++, GB-classic, RGGB-s, ScOrGBG-s
  python src/bench_r.py $ID results/bench_r/r_$ID.jsonl 3000 4    # CGB matched to each of them
done
python src/agg_r.py
```

`bench_r.py` reads `results/bench/b_<ID>.jsonl`, so `bench.py` must finish first. MNIST (554) and HAR (1478) can be run on a GPU with `python src/e9g.py <ID> <k>` (same protocol, double precision), where `k` = 0, 1, 2, 3 selects the generator whose ball count is matched (GBG++, GB-classic, RGGB-s, ScOrGBG-s); rows are appended to `e9g_results.jsonl`.

A row of `results/bench_r/r_11.jsonl` looks like this (values rounded here):

```json
{"did": 11, "dataset": "balance-scale", "n": 625, "d": 4, "method": "CGB-r", "matched_to": "GBG++",
 "lam": 1.7867, "n_balls": 227, "gen_sec": 0.4551,
 "order": 0.0, "jit0.01": 0.3385, "jit0.05": 0.8179, "sub95": 0.0718,
 "drift0.01": 3.0288, "drift0.05": 2.4693,
 "acc_rad": 0.8352, "acc_den": 0.7264, "acc": 0.8352, "cv_balls": 319.2, "sec": 21.6222}
```

| Field | Meaning |
|---|---|
| `n_balls`, `gen_sec` | number of balls and generation time on the full data set |
| `order` | 1 - ARI after a random permutation of the rows |
| `jit0.01`, `jit0.05` | 1 - ARI after Gaussian jitter with standard deviation 0.01 or 0.05 per coordinate |
| `sub95` | 1 - ARI after removing a random 5% of the samples, on the retained samples |
| `drift0.01`, `drift0.05` | movement of the assigned ball centers divided by the size of the jitter |
| `acc_rad`, `acc_den`, `acc` | 5-fold cross-validated accuracy with the radius rule, the density rule and the better of the two |
| `lam`, `matched_to` | fusion strength and the generator whose ball count it matches |

The rows of `results/bench/b_<ID>.jsonl` have the same fields for the greedy generators. Instabilities are averaged over 5 repetitions.

### 2. Other experiments on the 56 benchmark data sets

| Experiment | Command for one data set | Aggregation |
|---|---|---|
| Fusion strength by nested cross-validation | `python src/e19d.py $ID results/e19d/a_$ID.jsonl 3000 5` | `python src/agg_e19d.py` |
| GB-acc and GB-adp generators | `python src/e18.py $ID results/e18/a_$ID.jsonl 3000 2` | `python src/agg_e18.py`, `python src/agg_e18b.py` |
| Per-class k-means, Ward and k-center | `python src/e17.py $ID results/e17/a_$ID.jsonl 3000 4` | `python src/agg_e17.py` |
| Deterministic greedy generators and ablations | `python src/e11.py $ID results/e11/a_$ID.jsonl 3000 7` | `python src/agg_e11.py` |
| Shared prediction rules | `python src/e10.py $ID results/e10/a_$ID.jsonl 3000 4` | `python src/agg10.py results/e10` |
| Downstream GB learners | `python src/e12b.py $ID results/e12b/a_$ID.jsonl 3000 5` | `python src/agg_e12b.py` |

`e10.py`, `e11.py`, `e12b.py` and `e17.py` read the matched `lambda` values from `results/bench_r/` or the baselines from `results/bench/`, so run step 1 first. `e19d.py` and `agg_e19d.py` also need `results/bench/`.

### 3. Fusion-strength path, warm starts and certificate

```
mkdir -p results/e15 results/e3 results/e14
for ID in 11 15 32 37 54 182 1462 1480 1494 40975 40982 40994; do
  python src/e15.py $ID results/e15/l_$ID.jsonl 3000          # 16 values of lambda, cold and warm starts
done
python src/agg_e15.py

for ID in 15 37 54 1462 1464 1480 1494 1510 40975 40982 40984 40994 11 23 188; do
  python src/e3_cert.py $ID results/e3/c_$ID.jsonl 600 8      # certified and empirical breaking radius
  python src/e14.py     $ID results/e14/c_$ID.jsonl           # electrical-flow variant of the certificate
done
python src/agg_e14.py
```

### 4. Large data sets (GPU)

Covertype (OpenML 1596), MiniBooNE (41150) and HIGGS (23512); about 20,000 points are held out for testing.

```
# all generators, lambda matched to GBG++ at n = 25,000, training sizes 12,500 ... full
python src/e16w.py <ID> results/e16/<ID>.jsonl
# RGGB-s runs that need more than one hour or a large amount of memory (CPU only)
RGGB_CHUNKED=1 python src/e16r.py <ID> <n> <with_stab 0|1> results/e16r/<ID>.jsonl
# lambda chosen by 3-fold cross-validation on the first 25,000 training points
python src/e21.py cv  <ID> <lambda_index 0..12> results/e21/cv_<ID>_<index>.jsonl
python src/e21.py fit <ID> <n> <with_stab 0|1> "results/e21/cv_<ID>_*.jsonl" results/e21/fit_<ID>.jsonl
```

`e21b.py` runs the same computation split into one solver run per call (modes `cvf`, `fit`, `check`, `agg`), which is convenient on a job scheduler. Each row reports `sec`, `n_balls`, test accuracy (`acc_rad`, `acc_den`) and, with `with_stab=1`, `sub95` and `order`.

### 5. Image embeddings, streaming and differential privacy

```
python src/e7_cpu.py feat          # DINOv2 ViT-S/14 features of CIFAR-10 (downloaded on first use)
python src/e20.py 2                # stability of GBG++, GB-classic and CGB on 5,000 images
                                   # -> results/e7/e20_cifar10_radius.jsonl

# streaming: 20 batches of 200 points; last argument 1 shuffles the rows
python src/e8_stream.py 151 results/e8/s_151.jsonl 4000 20 4 0   # natural order: 151 1461 1471 1596
python src/e8_stream.py 182 results/e8/s_182.jsonl 4000 20 4 1   # shuffled:      28 32 182 1497
python src/agg_e8_exact.py

# differentially private granulation, 10 repetitions
# data sets: 6 28 32 151 182 554 1120 1461 1478 1486 1590 1596
python src/e5c.py <ID> results/e5c/d_<ID>.jsonl 10 4
python src/agg_e5c.py "results/e5c/*.jsonl" NCM GBG++ CGB CGB-sig
```

The arguments after the file pattern of `agg_e5c.py` are the method names as they appear in the result rows.

## Files

| File | Content |
|---|---|
| `src/cgb.py` | CPU solver: graph construction, dual FISTA, exact fusion read-out |
| `src/e9g.py` | GPU solver (PyTorch) and GPU benchmark runner |
| `src/baselines.py` | Greedy generators: GB-classic, GBG++, RGGB-s, ScOrGBG-s |
| `src/bench.py` | Data loading and the evaluation protocol |
| `src/bench_r.py` | Radius-graph CGB and matching of `lambda` to a target number of balls |
| `src/e2c_ablate.py` | Graph and weight variants (radius, uniform weights, nearest-neighbor graph, per-class) |
| `src/e3_cert.py`, `src/e14.py` | Certified radius |
| `src/e5_dp.py`, `src/e5c.py` | Differentially private granulation |
| `src/e7_cpu.py`, `src/e20.py` | CIFAR-10 embeddings and their stability |
| `src/e8_stream.py` | Streaming granulation |
| `src/e10.py` to `src/e21b.py` | Experiments listed above |
| `src/agg*.py` | Aggregation scripts that print the summary tables |

## License

MIT
