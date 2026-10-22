# Alternating optimizer pilot

Point-estimation diagnostic at n = 2,000, with the unchanged two-layer, 32-unit ELU network and 200 neural epochs. Each alternating cycle uses one neural epoch followed by a full-data beta step and gamma Newton fits. No bootstrap coverage was estimated.

Ten simulated datasets per cell (seeds 0–9), point-fit seed 0, and the same train/test split as the main simulation. Beta RMSE is the square root of the mean squared error over datasets and the eight coefficients. The joint-split arm uses 1,600 fitting rows and 400 validation rows for the index/link, then all rows for its final gamma fit. All other arms fit on the full 2,000 rows.

| Outcome | Link | Joint, 80/20 split | Joint, all rows | Alternating, all rows | Spline, all rows |
|---|---|---:|---:|---:|---:|
| continuous | linear | 0.02919 | 0.02245 | 0.02295 | 0.02275 |
| continuous | sfun | 0.01117 | 0.00973 | 0.00953 | 0.00955 |
| continuous | sigmoid | 0.01523 | 0.01305 | 0.01308 | 0.01286 |
| binary | linear | 0.06867 | 0.05447 | 0.05510 | 0.05598 |
| binary | sfun | 0.03064 | 0.02709 | 0.02682 | 0.02651 |
| binary | sigmoid | 0.03934 | 0.03581 | 0.03479 | 0.03495 |
| cox | linear | 0.03229 | 0.03027 | 0.03031 | 0.03010 |
| cox | sfun | 0.01701 | 0.01595 | 0.01524 | 0.01536 |
| cox | sigmoid | 0.02304 | 0.02220 | 0.02128 | 0.02063 |

Using all rows accounts for most of the RMSE improvement relative to the earlier split-based fit. Alternating training has similar RMSE to full-data joint Adam in this small pilot. Its RMSE ranges from 0.984 to 1.031 times SplinePLSI across these nine cells. Ten datasets are insufficient to establish equivalence or superiority.

For optimizer stability, each cell uses one fixed dataset (data seed 0) and eight optimizer seeds (0–7). Both methods use all rows, so there is no random holdout. The reported SD is the square root of the mean of the eight coordinate variances across optimizer seeds. It measures optimizer variation on this dataset, not sampling uncertainty.

| Outcome | Link | Joint optimizer SD | Alternating optimizer SD | Alternating / joint |
|---|---|---:|---:|---:|
| continuous | linear | 0.00369 | 0.00393 | 1.065 |
| continuous | sfun | 0.00339 | 0.00139 | 0.410 |
| continuous | sigmoid | 0.00424 | 0.00132 | 0.312 |
| binary | linear | 0.00544 | 0.00240 | 0.441 |
| binary | sfun | 0.00313 | 0.00085 | 0.272 |
| binary | sigmoid | 0.00393 | 0.00123 | 0.312 |
| cox | linear | 0.00483 | 0.00316 | 0.655 |
| cox | sfun | 0.00441 | 0.00064 | 0.145 |
| cox | sigmoid | 0.00446 | 0.00092 | 0.205 |

Across the nonlinear links, optimizer SD falls by 59–86%. Continuous linear is the exception to improved stability: its optimizer SD increases by about 6%. All 504 point fits completed without failure. The 90 joint-split beta estimates match the earlier matched-row diagnostic exactly. All 200 beta proposals were accepted in each alternating fit.

The full-data coefficient passes increase runtime. The CSV includes wall-clock timing under four concurrent diagnostic jobs, which includes CPU contention and should not be interpreted as a serial timing benchmark. `link_epochs` controls how frequently coefficients are updated. Cox network minibatches retain within-batch risk sets while coefficient updates use global risk sets.

Full bootstrap inference uses the same optimizer, regularization, data policy and epoch budget in every native C++ refit. A larger repeated-sampling coverage experiment is still needed for this changed estimator.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/compare_optimizers.py --n 2000 --replicates 10 \
  --optimizer_seeds 8 --jobs 4
```

`output/optimizer_comparison.json` contains the per-fit coefficients, diagnostics and summaries. `summary.csv` retains the aggregated pilot results, including gamma and link RMSE.
