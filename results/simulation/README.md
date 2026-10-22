# Simulation: NeuralPLSI vs SplinePLSI, n = 2000

**Interim snapshot (2026-10-22, 13:07 EDT).** The NeuralPLSI rerun (job
28702575) is still running. Everything here uses the datasets it has finished,
and the SplinePLSI numbers are restricted to the same seeds:

| g \ outcome | continuous | binary | cox |
|---|---:|---:|---:|
| linear | 500 | 500 | 500 |
| sfun (S-shape) | 389 | 500 | 500 |
| sigmoid | 500 | 500 | 500 |

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
  - A: coverage of the 95% interval for each coordinate of β (8) and γ (3), labelled with its true value. The shaded band is the Monte-Carlo range of one coordinate's coverage at exact 0.95.
  - B: mean bootstrap SE / empirical SD for each coordinate.
  - C: RMSE of each coordinate, and of g on |u| ≤ 2.
  - D: pointwise coverage of the 95% band for g.
- `inference.csv`: the numbers in panels A-C for each coordinate, plus block averages (`coordinate` = `all`, used for the ranges below). It also has `rmse_bagged`, the RMSE of the bagged estimate (the mean of the bootstrap refits).
- `summary_aggregated.csv`: bias (MAB), empirical SD, mean bootstrap SE and coverage for each β/γ block.
- `summary_table.csv` / `.tex`: the same quantities for each coefficient.
- `comparison_table_n2000_normal.csv` / `.tex`: per-coefficient bias, SD, SE and coverage, with the two estimators side by side.
- `comparison_metrics.csv`: the long-format data behind that table, including Monte-Carlo SEs.

## Main findings so far

Ranges are across the 9 settings. With 389-500 replicates, the Monte-Carlo
range at exact 0.95 coverage is ±0.022 to ±0.019 around 0.95 for a single
coefficient.

- **β intervals.**
  - NeuralPLSI covers 0.940-0.955; SplinePLSI covers 0.942-0.969.
  - SE/SD is 0.98-1.03 for both. The one exception is SplinePLSI linear-binary at 1.14, which over-covers (0.969), as in the previous run.
  - The previous NeuralPLSI run over-covered (0.967-0.984). That came from optimizer noise in the single fit. The bagged β is now no more accurate than the single fit (RMSE within 0.2%), so that noise is gone.
- **Per coordinate (panels A-C).**
  - NeuralPLSI's coverage is 0.922-0.968. Five of its 99 coordinates are below the Monte-Carlo range, about what chance alone gives, and none are above it. SE/SD is 0.91-1.09.
  - SplinePLSI's coverage is 0.918-0.980, with 12 of 99 outside the range. Five of those are linear-binary β coordinates that over-cover: SE/SD is 1.11-1.21 for all eight β coordinates there. Elsewhere its SE/SD is 0.93-1.09.
  - The lowest β coordinate for both is sigmoid-binary β7, a true zero, at 0.922. SplinePLSI's lowest γ coordinate is S-shape binary γ1 at 0.918. The two true zeros (β7, β8) otherwise behave like the other coordinates.
  - The per-coordinate RMSE ratio NeuralPLSI / SplinePLSI is 0.92-1.02. NeuralPLSI's largest gains are 6-8% on the linear-binary β coordinates.
- **β accuracy.** NeuralPLSI and SplinePLSI have the same RMSE to within 0.4% in 6 settings. NeuralPLSI is about 1% lower for linear-Cox and sigmoid-binary, and 6% lower for linear-binary. (In the previous run SplinePLSI was 11-16% lower.)
- **γ.**
  - Coverage is 0.933-0.945 for NeuralPLSI and 0.925-0.945 for SplinePLSI. SE/SD is 0.98-1.02 for both.
  - RMSE is equal or up to 3% lower for NeuralPLSI.
- **g, RMSE on |u| ≤ 2.**
  - NeuralPLSI is lower in 7 of 9 settings: 24-44% lower for linear g, 10-17% lower for sigmoid continuous and binary, 9% lower for S-shape binary and 4% lower for sigmoid Cox.
  - NeuralPLSI is about equal for S-shape Cox (0.115 vs 0.114) and 12% higher for S-shape continuous (0.060 vs 0.053).
- **g band, mean coverage on |u| ≤ 2.**
  - NeuralPLSI: 0.94-0.97 in 8 settings. The weak spot is S-shape binary, at 0.92 on |u| ≤ 2: it dips to about 0.83 near u = 0 and falls to about 0.59 at |u| = 3.
  - SplinePLSI: 0.92-0.97. It dips to about 0.85 at the steep part of the sigmoid for continuous and Cox, which is consistent with the bias of a 5-knot spline.
- **Prediction.** Test AUC and C-index agree to the third decimal. Test MSE for continuous outcomes is 1.008-1.011 for NeuralPLSI and 1.010-1.011 for SplinePLSI.
- **Cost (median per replicate, 40 threads).**
  - Fit: 0.3-1.1 s for NeuralPLSI; 0.07-0.3 s for SplinePLSI.
  - Bootstrap: 8-32 s for NeuralPLSI with B = 1000; 3-10 s for SplinePLSI with B = 999.
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
