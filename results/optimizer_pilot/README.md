# Alternating coefficient optimizer pilot

This pilot compares full-data joint AdamW, alternating coefficient/network updates,
and SplinePLSI. Both neural fits use the unchanged 2x32 ELU network, 200 neural
epochs, batch size 64, and no validation split. Alternating training uses one
coefficient cycle per neural epoch. No hyperparameters were selected from these
results.

Each of the nine existing simulation scenarios uses n=2000. Estimation error is
measured across eight datasets (seeds 0-7), with neural fit seed zero. Optimizer
stability uses one fixed dataset (seed zero) and eight fit seeds (0-7), varying
initialization and minibatch order. All 360 fits completed. Every method sees
exactly the same fitting rows within a comparison. These data use the same
2n generation and n-row train/test allocation as the simulation runner.

The reduction in optimizer variability is clear on these fixed datasets.
Across-dataset beta RMSE changed by -3.0% to +2.8% compared with full-data joint
AdamW. Thus this pilot supports stabilization, not a demonstrated efficiency
gain over full-data joint fitting. It does not isolate the contribution of the
beta step from the simultaneous move to full-data gamma updates each cycle.
Eight datasets per cell are insufficient for a firm performance conclusion,
and this is not a bootstrap coverage study.

## Beta RMSE across datasets

RMSE averages squared errors over all eight beta coordinates and datasets.
All three methods use all n fitting observations. These are new matched runs,
not the historical 500-replicate results with a neural holdout.

| Outcome | Link | Joint AdamW | Alternating | SplinePLSI |
|---|---|---:|---:|---:|
| continuous | linear | 0.02380 | 0.02448 | 0.02422 |
| continuous | sigmoid | 0.01383 | 0.01385 | 0.01380 |
| continuous | sfun | 0.01017 | 0.01018 | 0.01020 |
| binary | linear | 0.05525 | 0.05610 | 0.05747 |
| binary | sigmoid | 0.03652 | 0.03541 | 0.03545 |
| binary | sfun | 0.02768 | 0.02760 | 0.02719 |
| cox | linear | 0.03006 | 0.02978 | 0.02947 |
| cox | sigmoid | 0.02215 | 0.02154 | 0.02107 |
| cox | sfun | 0.01566 | 0.01527 | 0.01539 |

## Beta variability across optimizer seeds

Entries are the root mean of the eight coordinate-specific sample variances
across fit seeds, on a fixed dataset. This is optimization variability, not
sampling uncertainty.

| Outcome | Link | Joint AdamW SD | Alternating SD | Reduction |
|---|---|---:|---:|---:|
| continuous | linear | 0.00369 | 0.00348 | 5.6% |
| continuous | sigmoid | 0.00424 | 0.00119 | 71.9% |
| continuous | sfun | 0.00339 | 0.00140 | 58.8% |
| binary | linear | 0.00544 | 0.00239 | 56.1% |
| binary | sigmoid | 0.00393 | 0.00124 | 68.5% |
| binary | sfun | 0.00313 | 0.00085 | 72.8% |
| cox | linear | 0.00483 | 0.00325 | 32.7% |
| cox | sigmoid | 0.00446 | 0.00091 | 79.6% |
| cox | sfun | 0.00441 | 0.00066 | 85.1% |

Alternating training takes additional full-data derivative, line-search, and
gamma-fit passes. Concurrent median fit timings are recorded in `summary.json`
with the gamma and link-function error summaries. Four independent point fits
ran concurrently, so these wall times include CPU contention and are not an
isolated throughput benchmark. Native bootstrap remains inside C++.

Reproduce the pilot from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/benchmark_optimizers.py --datasets 8 --starts 8 --jobs 4 \
  --out output/optimizer_pilot.json
```

The output contains every fitted beta/gamma vector, errors, timings and fitting
diagnostics. `summary.json` preserves the source hash and aggregate results.
The source hash covers model and simulation code, not this report.
