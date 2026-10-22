# Simulation: NeuralPLSI vs SplinePLSI, n = 2000

Summaries of the current simulation run. The raw per-replicate results (about
940 MB of JSON) are not in git. They are on BigPurple at
`/gpfs/scratch/doh03/NeuralPLSI/output/simulation`.

| Setting | Value |
|---|---|
| Estimators | NeuralPLSI (commit `423b235`: full epoch budget, exact Newton γ, cold-start refits) and SplinePLSI (commit `816893d`: C++ profile fit) |
| Sample size | n = 2000 (training set; the test set has the same size) |
| g | linear, sfun (S-shape), sigmoid |
| Outcomes | continuous, binary, cox |
| Replicates | 500 per setting (seeds 0-499) |
| Bootstrap | B = 999 cold-start refits, 95% percentile intervals; both estimators use the same resamples |
| Exposures | normal |
| Run | 2026-10-21, `cpu_short`, 40 CPUs per task (job 28678283 for NeuralPLSI, 28680527 for SplinePLSI) |

The `source_sha256` recorded in every result row matches these commits.

## Files

- `inference.png` / `.pdf`, one figure with four panels:
  - A: coverage of the 95% intervals for β and γ. The shaded band is the Monte-Carlo range at exact 0.95 coverage.
  - B: mean bootstrap SE / empirical SD.
  - C: RMSE of β, γ and g. g is measured on |u| ≤ 2.
  - D: pointwise coverage of the 95% band for g.
- `inference.csv`: the numbers in panels A-C for each block. It also has `rmse_bagged`, the RMSE of the bagged estimate (the mean of the 999 refits).
- `summary_aggregated.csv`: bias (MAB), empirical SD, mean bootstrap SE and coverage for each β/γ block.
- `summary_table.csv` / `.tex`: the same quantities for each coefficient.
- `comparison_table_n2000_normal.csv` / `.tex`: per-coefficient bias, SD, SE and coverage, with the two estimators side by side.
- `comparison_metrics.csv`: the long-format data behind that table, including Monte-Carlo SEs.

## Main findings

Ranges are across the 9 settings unless stated. The Monte-Carlo range for 500
replicates at exact 0.95 coverage is 0.931-0.969.

- **β intervals.**
  - SplinePLSI covers 0.942-0.969.
  - NeuralPLSI covers 0.967-0.984. That is above 0.95 in every setting and above the Monte-Carlo range in 7 of 9.
  - SE/SD is 0.98-1.03 for both estimators. The one exception is SplinePLSI linear-binary at 1.14. That is systematic (median ratio 1.13), not caused by outliers.
- **Why NeuralPLSI's β intervals over-cover.**
  - The intervals have the right width: about 2 × 1.96 × the SD of β̂, and the errors are close to normal.
  - Percentile intervals are centred on the bootstrap distribution. The refits average out the optimizer noise of the original fit, so that centre is closer to the truth than β̂ is.
  - The bagged β has 11-19% lower RMSE than the single fit. That is the same as SplinePLSI to within 2%, and 5% lower for linear-binary.
  - Intervals of the same width centred on β̂ cover 0.94-0.95. (This check used the raw results and is not in the CSV.)
- **β accuracy.** For the single fit, SplinePLSI's β RMSE is 11-16% lower than NeuralPLSI's. The bagged NeuralPLSI β closes that gap.
- **γ.**
  - Both estimators are well calibrated: SE/SD is 0.97-1.03. Coverage is 0.935-0.955 for NeuralPLSI and 0.925-0.945 for SplinePLSI.
  - Binary S-shape is the lowest for both (0.935 and 0.925).
  - RMSE is equal for continuous and binary. SplinePLSI is 5-7% lower for Cox.
- **g, RMSE on |u| ≤ 2.**
  - SplinePLSI is lower for continuous (0.053-0.057 vs 0.064-0.084) and Cox (0.087-0.114 vs 0.099-0.144).
  - NeuralPLSI is lower for binary (0.138-0.214 vs 0.159-0.228).
- **g band.**
  - NeuralPLSI is mostly conservative (0.95-1.0) on |u| ≤ 2. It under-covers in the sparse tails for S-shape, down to about 0.65 at |u| = 3 for binary and 0.8 for Cox.
  - SplinePLSI is near 0.95, but dips to about 0.85 at the steep part of the sigmoid for continuous and Cox. This is consistent with the bias of a 5-knot spline.
- **Prediction.** Test AUC and C-index agree to the third decimal. Test MSE for continuous outcomes is 1.010-1.011 for SplinePLSI and 1.012-1.017 for NeuralPLSI.
- **Cost (median per replicate, 40 threads).**
  - Fit: about 0.5 s for NeuralPLSI; 0.07-0.3 s for SplinePLSI.
  - 999-refit bootstrap: 14-19 s for NeuralPLSI; 3-10 s for SplinePLSI.

## Regenerate

```bash
python reproduce/plot_inference.py --input output/simulation --out results/simulation
python reproduce/visualize_simulation.py --input output/simulation --out results/simulation --no-plots
python reproduce/table_simulation_comparison.py --input output/simulation --out results/simulation
```

`visualize_simulation.py` also writes per-setting tables to `tables/`. They are
not kept here because `summary_table.csv` has the same numbers.
