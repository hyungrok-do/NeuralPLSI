# Simulation: NeuralPLSI vs SplinePLSI, n = 2000

**Interim snapshot (2026-10-22, 10:50 EDT).** The NeuralPLSI rerun (job
28702575) is still running. Everything here uses the datasets it has finished,
and the SplinePLSI numbers are restricted to the same seeds:

| g \ outcome | continuous | binary | cox |
|---|---:|---:|---:|
| linear | 318 | 500 | 500 |
| sfun (S-shape) | 143 | 277 | 291 |
| sigmoid | 211 | 263 | 341 |

The raw per-replicate results are not in git. They are on BigPurple at
`/gpfs/scratch/doh03/NeuralPLSI/output/simulation`.

| Setting | Value |
|---|---|
| Estimators | NeuralPLSI (commit `d7338d0`: full-batch L-BFGS, exact Newton γ, no validation split) and SplinePLSI (commit `816893d`: C++ profile fit) |
| Sample size | n = 2000 (training set; the test set has the same size) |
| g | linear, sfun (S-shape), sigmoid |
| Outcomes | continuous, binary, cox |
| Replicates | 500 per setting (seeds 0-499) when complete; see the table above |
| Bootstrap | Cold-start refits, 95% percentile intervals. B = 1000 for NeuralPLSI; B = 999 for SplinePLSI, which was not rerun |
| Exposures | normal |
| Run | NeuralPLSI: 2026-10-22, job 28702575. SplinePLSI: 2026-10-21, job 28680527. Both on `cpu_short` with 40 CPUs per task |

The `source_sha256` recorded in every result row matches these commits.

## Files

- `inference.png` / `.pdf`, one figure with four panels:
  - A: coverage of the 95% intervals for β and γ. The shaded band is the Monte-Carlo range at exact 0.95 coverage.
  - B: mean bootstrap SE / empirical SD.
  - C: RMSE of β, γ and g. g is measured on |u| ≤ 2.
  - D: pointwise coverage of the 95% band for g.
- `inference.csv`: the numbers in panels A-C for each block. It also has `rmse_bagged`, the RMSE of the bagged estimate (the mean of the bootstrap refits).
- `summary_aggregated.csv`: bias (MAB), empirical SD, mean bootstrap SE and coverage for each β/γ block.
- `summary_table.csv` / `.tex`: the same quantities for each coefficient.
- `comparison_table_n2000_normal.csv` / `.tex`: per-coefficient bias, SD, SE and coverage, with the two estimators side by side.
- `comparison_metrics.csv`: the long-format data behind that table, including Monte-Carlo SEs.

## Main findings so far

Ranges are across the 9 settings. With 143-500 replicates, the Monte-Carlo
range at exact 0.95 coverage is ±0.036 to ±0.019 around 0.95 for a single
coefficient.

- **β intervals.**
  - NeuralPLSI covers 0.938-0.957; SplinePLSI covers 0.939-0.969.
  - SE/SD is 0.97-1.03 for both. The one exception is SplinePLSI linear-binary at 1.14, which over-covers (0.969), as in the previous run.
  - The previous NeuralPLSI run over-covered (0.967-0.984). That came from optimizer noise in the single fit. The bagged β is now no more accurate than the single fit (RMSE within 0.3%), so that noise is gone.
- **β accuracy.** NeuralPLSI and SplinePLSI have the same RMSE to within 0.3% in 6 settings. NeuralPLSI is about 1% lower for linear-Cox and sigmoid-binary, and 6% lower for linear-binary. (In the previous run SplinePLSI was 11-16% lower.)
- **γ.**
  - Coverage is 0.935-0.952 for NeuralPLSI and 0.924-0.954 for SplinePLSI. SE/SD is 0.96-1.05 for both.
  - RMSE is equal or up to 3% lower for NeuralPLSI.
- **g, RMSE on |u| ≤ 2.**
  - NeuralPLSI is lower in 7 of 9 settings: 23-44% lower for linear g, 10-18% lower for sigmoid continuous and binary, 11% lower for S-shape binary and 5% lower for sigmoid Cox.
  - The two are equal for S-shape Cox (0.116). NeuralPLSI is 12% higher for S-shape continuous (0.059 vs 0.053).
- **g band, mean coverage on |u| ≤ 2.**
  - NeuralPLSI: 0.94-0.96 in 8 settings. The weak spot is S-shape binary, at 0.92 on |u| ≤ 2: it dips to about 0.82 near u ≈ 0.2 and falls to about 0.57 at |u| = 3.
  - SplinePLSI: 0.92-0.97. It dips to about 0.84-0.85 at the steep part of the sigmoid for continuous and Cox, which is consistent with the bias of a 5-knot spline.
- **Prediction.** Test AUC and C-index agree to the third decimal. Test MSE for continuous outcomes is 1.005-1.007 for NeuralPLSI and 1.006-1.009 for SplinePLSI.
- **Cost (median per replicate, 40 threads).**
  - Fit: 0.3-1.1 s for NeuralPLSI; 0.07-0.3 s for SplinePLSI.
  - Bootstrap: 8-34 s for NeuralPLSI with B = 1000; 3-16 s for SplinePLSI with B = 999.
  - NeuralPLSI is slowest for continuous outcomes, and S-shape continuous is the slowest setting.

## Regenerate

When the run is complete:

```bash
python reproduce/plot_inference.py --input output/simulation --out results/simulation
python reproduce/visualize_simulation.py --input output/simulation --out results/simulation --no-plots
python reproduce/table_simulation_comparison.py --input output/simulation --out results/simulation
```

For this snapshot, `--input` was a folder holding the partial NeuralPLSI files
and the SplinePLSI files filtered to the same seeds.

`visualize_simulation.py` also writes per-setting tables to `tables/`. They are
not kept here because `summary_table.csv` has the same numbers.
