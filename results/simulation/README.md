# Simulation: NeuralPLSI fitted by L-BFGS and by minibatch AdamW, and SplinePLSI, n = 2000

Summaries of the current simulation run. The raw per-replicate results are not
in git. They are on BigPurple at `/gpfs/scratch/doh03/NeuralPLSI/output/simulation`.

| Setting | Value |
|---|---|
| Estimators | NeuralPLSI (full-batch L-BFGS, exact Newton γ, no validation split), SplinePLSI (C++ profile fit), NeuralPLSI-AdamFrozen (minibatch AdamW with early stopping on a 20% validation split, then the frozen correction: Newton steps on β, γ and the intercept with the network fixed, on all rows), and NeuralPLSI-AdamGamma, the same AdamW fit followed by the exact γ fit only (the comparator: β is not corrected) |
| Sample size | n = 2000 (training set; the test set has the same size) |
| g | linear; sigmoid; threshold, g(u) = 2 tanh(1.5 max(u, 0)) + 0.1u, nearly flat below u = 0 and then a steep rise that levels off, with a kink at u = 0 |
| Outcomes | continuous, binary, cox |
| Replicates | 500 per setting (seeds 0-499) |
| Bootstrap | Cold-start refits, 95% percentile intervals. B = 1000, except SplinePLSI with linear and sigmoid g (B = 999), which was not rerun |
| Exposures | normal |
| Code | Linear and sigmoid g: NeuralPLSI commit `d7338d0`, SplinePLSI `816893d`, NeuralPLSI-AdamGamma `99e1a1c`. Threshold g: these three at commit `2a11768`, where its key was `nlrelu`; the raw files were relabelled `threshold` afterwards, with no other change. NeuralPLSI-AdamFrozen, all g: commit `0ba060d` |
| Run | Linear and sigmoid g: NeuralPLSI 2026-10-22, job 28702575; NeuralPLSI-AdamGamma 2026-10-23, job 28718391; SplinePLSI 2026-10-21, job 28680527. Threshold g: these three 2026-10-23 to 24, job 28756719. NeuralPLSI-AdamFrozen 2026-10-27, job 28865591. All on `cpu_short` with 40 CPUs per task. NeuralPLSI's fitting code is unchanged since, so its rows stand |

The `source_sha256` recorded in every result row matches these commits.

## Files

- `inference.png` / `.pdf`, one figure with four panels and the four estimators (NeuralPLSI blue, SplinePLSI orange, NeuralPLSI-AdamFrozen green, NeuralPLSI-AdamGamma purple, which panel D leaves out because neither correction changes the network, so its g is NeuralPLSI-AdamFrozen's). Panels A and B are forest plots, one row per coordinate of β (8) and γ (3), labelled with its true value:
  - A: the mean bias of each coordinate in units of its empirical SD, with its 95% Monte-Carlo interval over the 500 replicates (±0.088 SD). The shaded band is the Monte-Carlo range at exactly zero bias. The axis stops at ±0.5 SD; a bias beyond it is drawn as an arrow labelled with its value.
  - B: coverage of the 95% interval for each coordinate, with its 95% Monte-Carlo (Wilson) interval. The shaded band is the Monte-Carlo range of one coordinate's coverage at exact 0.95.
  - C: mean bootstrap SE / empirical SD for each coordinate.
  - D: the estimated g, its mean over the 500 replicates with a band of ±1 empirical SD, against the true g. The shading marks |u| > 2, which holds 7% of the index values.
- `inference.csv`: the numbers in panels A-C for each coordinate, plus block averages (`coordinate` = `all`, used for the ranges below): `bias` with its Monte-Carlo half-width `bias_mc` and `std_bias` (bias / empirical SD; block rows average their absolute values), coverage, SE/SD, and the RMSE of each coordinate and of g on |u| ≤ 2. It also has `rmse_bagged`, the RMSE of the bagged estimate (the mean of the bootstrap refits).
- `summary_aggregated.csv`: bias (MAB), empirical SD, mean bootstrap SE and coverage for each β/γ block.
- `summary_table.csv` / `.tex`: the same quantities for each coefficient.
- `comparison_table_n2000_normal.csv` / `.tex`: per-coefficient bias, SD, SE and coverage, with the estimators side by side.
- `comparison_metrics.csv`: the long-format data behind that table, including Monte-Carlo SEs.

## Main findings: NeuralPLSI and SplinePLSI

Ranges are across the 9 settings. The Monte-Carlo range for 500 replicates at
exact 0.95 coverage is 0.931-0.969 for a single coefficient.

- **β intervals.**
  - NeuralPLSI covers 0.940-0.971; SplinePLSI covers 0.942-0.977.
  - For linear and sigmoid g, SE/SD is 0.98-1.03 for both. The one exception is SplinePLSI linear-binary at 1.14, which over-covers (0.969), as in the previous run.
  - For threshold g both are conservative with binary outcomes: SE/SD is 1.15 for NeuralPLSI (coverage 0.971) and 1.34 for SplinePLSI (0.977). NeuralPLSI is also slightly conservative with continuous outcomes (1.08, coverage 0.963). The excess is spread across the replicates (median per-replicate SE/SD 1.11 and 1.32 for binary), not caused by a few unstable refits. Its cause is not yet known; it appears only with the kinked link.
  - An earlier NeuralPLSI run over-covered (0.967-0.984) because of optimizer noise in the single fit. The bagged β is now no more accurate than the single fit (RMSE within 0.6%), so that noise is gone.
- **Bias (panel A).** Both estimators have small finite-sample biases: at most 0.25 SD for NeuralPLSI and 0.20 SD for SplinePLSI (at most 0.012 on the scale of the coefficients). With 500 replicates a bias of 0.09 SD is detectable, so about a quarter of the coordinates fall outside the Monte-Carlo range for zero (25 of 99 for each), most of them for binary outcomes and for γ with Cox outcomes. A bias of 0.25 SD lowers the coverage of a 95% interval to about 0.94.
- **Per coordinate (panels B and C; RMSE from `inference.csv`).**
  - NeuralPLSI's coverage is 0.922-0.986. For linear and sigmoid g it is 0.922-0.968, with 3 of 66 coordinates below the Monte-Carlo range, about what chance alone gives, and none above it; SE/SD is 0.91-1.09. For threshold g, 8 of 33 are above the range (6 binary, 2 continuous), with SE/SD up to 1.22.
  - SplinePLSI's coverage is 0.920-0.988. For linear and sigmoid g, 9 of 66 are outside the range: 4 below, and 5 linear-binary β coordinates above (SE/SD is 1.11-1.21 for all eight β coordinates there). Elsewhere its SE/SD is 0.93-1.09. For threshold g, 10 of 33 are above the range: all eight binary β coordinates (SE/SD 1.18-1.52), binary γ3 and continuous β8.
  - The lowest β coordinate for both is sigmoid-binary β7, a true zero, at 0.922. The lowest γ coordinate for both is sigmoid-binary γ3 (0.926 and 0.920). The two true zeros (β7, β8) otherwise behave like the other coordinates.
  - The per-coordinate RMSE ratio NeuralPLSI / SplinePLSI is 0.88-1.01. NeuralPLSI's largest gains are on the β coordinates of threshold g (4-12% for continuous outcomes, 5-8% for Cox, 3-7% for binary) and of linear-binary (4-8%).
- **β accuracy.** NeuralPLSI and SplinePLSI have the same RMSE to within 0.4% for linear-continuous and sigmoid continuous and Cox. NeuralPLSI is about 1% lower for linear-Cox and sigmoid-binary, 6% lower for linear-binary, and 5-8% lower for all three threshold settings.
- **γ.**
  - Coverage is 0.939-0.954 for NeuralPLSI and 0.931-0.958 for SplinePLSI. SE/SD is 0.98-1.05 for both.
  - RMSE is equal or up to 2% lower for NeuralPLSI.
- **g, RMSE on |u| ≤ 2.** NeuralPLSI is lower in all 9 settings: 24-44% lower for linear g, 4-17% lower for sigmoid g and 10-34% lower for threshold g (34% for continuous outcomes, 24% for Cox, 10% for binary).
- **g, bias and spread (panel D).** On |u| ≤ 2 both follow the true g closely, except at the kink of threshold g, which both round off. That setting has the largest bias for every outcome.
  - NeuralPLSI's largest pointwise bias is 0.006-0.069 for continuous outcomes, 0.026-0.143 for binary and 0.012-0.228 for Cox. Its spread (mean SD over the replicates) is 0.038-0.055, 0.077-0.123 and 0.047-0.090.
  - SplinePLSI's largest bias is 0.007-0.230, 0.048-0.190 and 0.021-0.388, and its spread is larger for binary outcomes and for linear g, and similar otherwise (0.051-0.053, 0.146-0.157 and 0.076-0.083). Its 5 knots round the kink more: for continuous and binary outcomes its largest bias is at the kink itself (0.23 and 0.19 at u = 0).
  - For Cox outcomes the level of g is not identified, and both estimators fix it by setting g(0) = 0. Threshold g has its kink exactly there, so rounding off the kink raises the fitted curve at u = 0, and the anchor then moves the whole curve down. The mean bias on |u| ≤ 2 is -0.17 for NeuralPLSI and -0.24 for SplinePLSI. That offset is most of the largest bias (0.23 and 0.39; 0.16 and 0.23 once a constant is removed) and of the Cox g RMSE. It comes from placing the kink at the anchor, not from either estimator.
  - Beyond |u| = 2 the binary fits have little information. At |u| = 3, NeuralPLSI stays near the true g (bias at most 0.28, SD 0.26-0.46), while SplinePLSI spreads widely (SD 0.60-1.75) and overshoots for threshold g (bias 0.83 at u = 3).
- **Prediction.** Test AUC and C-index differ by at most 0.0011. Test MSE for continuous outcomes is 1.008-1.011 for NeuralPLSI and 1.010-1.018 for SplinePLSI, whose highest is for threshold g.
- **Cost (median per replicate, 40 threads).**
  - Fit: 0.3-0.8 s for NeuralPLSI; 0.07-0.4 s for SplinePLSI.
  - Bootstrap: 8-25 s for NeuralPLSI with B = 1000; 3-10 s for SplinePLSI.
  - NeuralPLSI is slowest for continuous outcomes (bootstrap 16-25 s).

## Minibatch AdamW with the frozen correction (NeuralPLSI-AdamFrozen)

The network is trained the way neural networks usually are: AdamW on
minibatches of 64 rows from a random 80% of the data, early stopping on the
other 20% (patience 10, at most 200 epochs) with the best checkpoint restored.
Then, on all rows, the frozen correction takes Newton steps on β (in the
tangent plane of the unit sphere), γ and the intercept with the network fixed,
until they converge. Every bootstrap refit uses the same procedure from a fresh
initialization. Ranges are across the 9 settings and compare it with L-BFGS.

- **β: as accurate as L-BFGS, and its intervals cover.** β RMSE is 0-3% above
  L-BFGS's. Coverage is 0.941-0.968 for the block (L-BFGS 0.940-0.971), with 4
  of 72 coordinates below the Monte-Carlo range and 6 above (L-BFGS 2 and 7);
  SE/SD is 0.97-1.14, the highest for threshold-binary, as for L-BFGS (1.15).
- **γ: as accurate as L-BFGS, and its intervals cover.** RMSE is 1% below to 9%
  above L-BFGS's (the most for Cox). Coverage is 0.943-0.964 with SE/SD
  0.98-1.05, none of 27 coordinates below the Monte-Carlo range.
- **g is AdamW's**: the correction does not change the network, so g's RMSE is
  4-88% above L-BFGS's, the most for linear g.
- **Prediction.** Test MSE is 1.010-1.016 (L-BFGS 1.008-1.011); AUC and C-index
  match L-BFGS's to 0.001.
- **Cost.** Fits take 0.28-0.31 s and the bootstrap 6-9 s with B = 1000
  (L-BFGS: 0.3-0.8 s and 8-25 s).

## AdamW followed by the exact γ fit only (NeuralPLSI-AdamGamma)

The comparator runs the same AdamW fit and then fits only γ and the intercept
exactly on all rows, with the index and link fixed. Its g is
NeuralPLSI-AdamFrozen's (the same network), but its β is AdamW's, uncorrected.

- **β: intervals wider than needed, estimates less accurate.** Coverage is
  0.965-0.982, against 0.941-0.968 with the frozen correction, and 41 of 72
  coordinates are above the Monte-Carlo range: the bootstrap's percentile
  interval is centred on the bagged estimate, which averages out AdamW's
  optimizer noise in the single fit. β RMSE is 10-15% above L-BFGS's.
- **γ is fine**: coverage 0.943-0.965, SE/SD 0.98-1.05, RMSE 1% below to 10%
  above L-BFGS's. Without the γ step (plain AdamW, an earlier arm) early
  stopping biased γ₁ toward zero by 1.1-1.4 empirical SDs and its coverage fell
  to 0.89-0.94.
- **Cost.** The bootstrap takes 5-7 s with B = 1000.

So for minibatch training the correction on all rows is what makes the
bootstrap work: correcting γ alone fixes γ, and correcting β as well makes β as
accurate as full-batch L-BFGS, at a lower cost.

## Regenerate

```bash
python reproduce/plot_inference.py --input output/simulation --out results/simulation
python reproduce/visualize_simulation.py --input output/simulation --out results/simulation --no-plots
python reproduce/table_simulation_comparison.py --input output/simulation --out results/simulation
```

`visualize_simulation.py` also writes per-setting tables to `tables/`. They are
not kept here because `summary_table.csv` has the same numbers.
