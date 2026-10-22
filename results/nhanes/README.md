# NHANES: NeuralPLSI vs SplinePLSI

Model: `g(X'beta) + Z'gamma`, `||beta|| = 1`, `beta[0] >= 0`. n = 800 complete
observations, p = 8 exposures (log-transformed except retinol, then z-scored),
q = 6 covariates (age, sex, four race indicators) plus an intercept. Outcome:
z-scored log triglyceride, so coefficients are in SD units. Unweighted.

| Setting | NeuralPLSI | SplinePLSI |
|---|---|---|
| Fit | full-batch L-BFGS on all rows, 2x32 ELU network, L2 penalty 1e-3, exact gamma | 5 knots, cubic B-spline, ridge 1e-6 |
| Bootstrap | B = 1000 cold-start pairs refits, 95% percentile intervals | B = 1000, same procedure |
| Code | commit `d7338d0` | same commit |
| Time (local, 10 threads) | fit 0.4 s; bootstrap 89 s | fit 0.05 s; bootstrap 8 s |

## Findings

- The index directions agree: cos(beta_NN, beta_spline) = 0.999 (2.7 degrees).
- All 8 exposure loadings exclude 0 under both models. alpha-tocopherol (+0.69)
  and gamma-tocopherol (+0.42) load positively, trans-beta-carotene (-0.35),
  PCB 194 (-0.27) and HxCDF (-0.27) negatively.
- NeuralPLSI's beta intervals are narrower on average (mean width 0.165 vs 0.180),
  most for retinyl stearate (bootstrap SE 0.048 vs 0.107). Covariate intervals
  are the same width (0.291 vs 0.293); 5 of 6 exclude 0 for both, all but sex.
- Both links rise monotonically and almost linearly over the bulk of the index,
  with mild convexity. The spline turns up at the lower edge and its band widens
  sharply above index 3, where data are sparse; the neural curve stays smooth.

The g-function plots show the bootstrap mean of g plus the intercept, with its
joint 95% band, over partial residuals. The forest plot sorts coefficients by
the size of the NeuralPLSI estimate.

## Files

- `nhanes_comparison.csv`: both models' estimates, SEs and 95% intervals for beta and gamma.
- `nhanes_nplsi_results.csv`: NeuralPLSI only, including the intercept.
- `nhanes_g_function_neural.png`, `nhanes_g_function_plsi.png`, `nhanes_forest_plot.png`.

Regenerate (needs `NHANES/12940_2020_644_MOESM2_ESM.csv`):

```bash
python reproduce/nhanes_comparison.py   # comparison CSV and figures in output/nhanes
python reproduce/nhanes.py              # NeuralPLSI-only CSV in output/
```
