# NeuralPLSI simulation: native C++ backend, B = 999

Summaries of the first full coverage run of the C++ NeuralPLSI (commit `7270f3c`).
The raw per-replicate results (about 900 MB of JSON) stay out of git. They are on
BigPurple at `/gpfs/scratch/doh03/NeuralPLSI/output/simulation_native_v4_b999`.

| Setting | Value |
|---|---|
| Model | NeuralPLSI only (array IDs 0-8) |
| Sample sizes | n = 1000 and 2000 (training set; the test set has the same size) |
| g | linear, sfun (S-shape), sigmoid |
| Outcomes | continuous, binary, cox |
| Replicates | 500 per setting (seeds 0-499) |
| Bootstrap | B = 999 draws, warm-start refits, 95% percentile intervals |
| Exposures | normal |
| Run | 2026-10-21, `cpu_short`, 20 CPUs per task (jobs 28661624 and 28661625) |

## Files

- `inference.png` / `.pdf`, one figure with four panels:
  - A: coverage of the 95% intervals for β and γ.
  - B: bootstrap SE / empirical SD.
  - C: pointwise coverage of the g band.
  - D: fits that stopped early in the first training phase (joint epochs < 15).
- `summary_aggregated.csv`: bias (MAB), empirical SD, mean bootstrap SE and coverage for each β/γ block and setting.
- `summary_table.csv` / `.tex`: the same quantities for each coefficient.
- `early_stopping.csv`: number of fits that stopped before epoch 15 in each setting.

## Main findings

- The β intervals under-cover in every setting. β coverage is 0.71-0.93 at n = 1000 and 0.75-0.97 at n = 2000.
  The bootstrap SE for β is 0.33-0.92 times the empirical SD. γ is well calibrated: coverage is 0.95-1.00 and SE/SD is about 1.
- Some fits stop at joint epoch 7-8, giving a near-flat g and a β error of about 1.
  This happens in 99 of 500 linear-binary replicates at n = 1000 and at most 16 elsewhere (1 in total at n = 2000).
  These fits widen the intervals, so they *raise* the reported β coverage.
- The g band under-covers where the curvature is high (for example S-shape continuous near u = ±1.5),
  as well as in the sparse tails (|u| > 2). Coverage at u = 0 is 1 by construction, because g(0) = 0 is fixed for identification.

## Regenerate

```bash
python reproduce/visualize_simulation.py --input output/simulation_native_v4_b999 --no-plots   # summary_*.csv/.tex
python reproduce/plot_native_inference.py --input output/simulation_native_v4_b999
```
