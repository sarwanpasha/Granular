# Granular

Convex Granular Balls (CGB): granular-ball generation as a strongly convex, label-aware fusion problem on a radius graph.

Each sample gets a representative, and representatives of same-label samples closer than a radius `r` are pulled together by a weighted sum of norms. The minimizer is unique, so the balls (the connected components of the exactly fused edges) do not depend on the order of the rows or on random seeds. This repository contains the CPU and GPU solvers, the baseline generators and the scripts for all experiments.

## Installation

```
pip install -r requirements.txt
```

The GPU solver (`src/e9g.py`) needs a CUDA build of PyTorch. Data sets are downloaded from OpenML on first use into `./data`.

## Layout

| File | Content |
|---|---|
| `src/cgb.py` | CPU solver: dual FISTA with exact fusion read-out |
| `src/e9g.py` | GPU solver (PyTorch) and the GPU benchmark runner |
| `src/baselines.py` | GB-classic, GBG++, RGGB-s and ScOrGBG-s generators |
| `src/bench.py` | Data loading and the evaluation protocol (order, jitter, subsampling, drift, cross-validated accuracy) |
| `src/bench_r.py` | Radius-graph CGB and matching of the fusion strength to a target number of balls |
| `src/e2c_ablate.py` | Graph and weight variants used in the ablations |
| `src/e3_cert.py`, `src/e14.py` | Certified radius and its electrical-flow variant |
| `src/e10.py` | Accuracy with shared prediction rules |
| `src/e11.py` | Deterministic greedy generators and ablations |
| `src/e12b.py` | Downstream granular-ball learners |
| `src/e15.py` | Fusion-strength path and warm starts |
| `src/e17.py` | Per-class k-means, Ward and k-center baselines |
| `src/e18.py` | Accelerated and adaptive granular-ball generators (GB-acc, GB-adp) |
| `src/e19d.py` | Fusion strength chosen by nested cross-validation |
| `src/e7_cpu.py`, `src/e20.py` | DINOv2 embeddings of CIFAR-10 and their stability |
| `src/e8_stream.py` | Streaming granulation |
| `src/e5_dp.py`, `src/e5c.py` | Differentially private granulation |
| `src/e16w.py`, `src/e16r.py`, `src/e21.py`, `src/e21b.py` | Large-scale runs (Covertype, MiniBooNE, HIGGS) |
| `src/agg*.py` | Aggregation of results into medians, win/tie/loss counts and Holm-adjusted Wilcoxon tests |

Every script documents its command-line usage in its docstring. Scripts write their results to `./results`.

## License

MIT
