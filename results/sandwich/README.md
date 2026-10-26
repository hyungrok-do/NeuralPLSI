# Sandwich standard errors for NeuralPLSI and SplinePLSI

A full-parameter sandwich SE for β and γ, tested against the truth in
simulation. It needs no refitting: one Hessian and one score covariance at the
fit.

## Method

NeuralPLSI's fit minimizes the penalized objective

    Q(θ) = (1/n) Σ_i l_i(θ) + g(0)² + (λ/2) |network|²

over all parameters θ: β, γ, the skip weight, every weight and bias of the
network and the intercept (1164 parameters for 8 exposures, 3 confounders and a
2 × 32 network). At a minimum of Q, the M-estimator sandwich is

    Var(θ̂) ≈ H⁻¹ C H⁻¹ / n,   H = ∇²Q(θ̂),   C = Cov(∇l_i(θ̂)).

β is identified only up to scale, so it enters as coordinates d in the tangent
plane of the unit sphere at β̂ (β = β̂ + T d, T an orthonormal basis of the
complement of β̂), and Var(β̂) = T Var(d̂) Tᵀ. Inverting the full Hessian carries
the estimation of the network into the variance of β and γ.

`reproduce/sandwich/sandwich.py` writes Q in PyTorch, with the engine's layout
and loss, and differentiates it exactly. At the fitted weights it matches the
engine's loss to 1e-7 and its gradient to 5e-7. Continuous and binary outcomes
only: the Cox partial likelihood is not a sum of independent terms and needs the
score-residual form, not built yet.

## Monte Carlo design

Continuous and binary outcomes with linear, sigmoid and threshold g; n = 2000
and the data of the main simulation (seeds 0-199, so 200 datasets per setting);
one fit per dataset (fit seed 1337); 2 × 32 ELU network. SE/SD is the mean
sandwich SE over the datasets divided by the empirical SD of the estimates;
coverage is that of β̂ ± 1.96 SE. Both are averaged over the 8 coordinates of β
or the 3 of γ. With 200 datasets the Monte-Carlo range of one coordinate's
coverage at 0.95 is 0.920-0.980.

Four fits:

- **Converged L-BFGS**: the package's L-BFGS with a strict stopping rule (relative
  decrease below 1e-9 over 20 iterations; `build_converged.sh`).
- **Package stopping rule**: L-BFGS as the package runs it (1e-5 over 10).
- **AdamW**: the minibatch AdamW baseline of the main simulation, with early
  stopping on a 20% validation split. Its penalty is its weight decay, 1e-4.
- **AdamW + exact γ**: the same fit, then γ fitted exactly with the network fixed.

## Results

SE/SD / coverage, from `sandwich_monte_carlo.csv`:

| setting | converged L-BFGS β | γ | package rule β | γ | AdamW β | γ | AdamW + exact γ β | γ |
|---|---|---|---|---|---|---|---|---|
| continuous, linear | 1.02 / 0.951 | 0.96 / 0.940 | 1.03 / 0.952 | 0.96 / 0.940 | 0.56 / 0.649 | 0.89 / 0.812 | 0.50 / 0.654 | 0.98 / 0.942 |
| continuous, sigmoid | 1.11 / 0.956 | 0.97 / 0.943 | 1.09 / 0.958 | 0.97 / 0.942 | 0.52 / 0.646 | 0.80 / 0.780 | 0.53 / 0.644 | 1.04 / 0.945 |
| continuous, threshold | 1.13 / 0.956 | 0.99 / 0.942 | 1.42 / 0.962 | 1.04 / 0.940 | 0.70 / 0.644 | 0.88 / 0.788 | 0.54 / 0.642 | 0.97 / 0.940 |
| binary, linear | 1.00 / 0.955 | 1.05 / 0.960 | 1.05 / 0.956 | 1.05 / 0.962 | 0.63 / 0.696 | 1.02 / 0.847 | 0.55 / 0.689 | 1.05 / 0.965 |
| binary, sigmoid | 1.04 / 0.955 | 1.06 / 0.960 | 1.07 / 0.955 | 1.06 / 0.965 | 0.54 / 0.681 | 0.90 / 0.803 | 2.03 / 0.682 | 2.98 / 0.958 |
| binary, threshold | 1.09 / 0.946 | 1.02 / 0.963 | 1.19 / 0.947 | 1.03 / 0.965 | 1.26 / 0.762 | 1.08 / 0.843 | 0.74 / 0.756 | 1.05 / 0.952 |

- **Converged L-BFGS: valid.** β covers 0.946-0.956 and γ 0.940-0.963 in every
  setting; 65 of the 66 coordinates are inside the Monte-Carlo range, and one (a γ
  coordinate, continuous sigmoid, 0.915) just below it, about what chance alone gives. SE/SD is
  1.00-1.13 for β (slightly conservative for curved g with continuous outcomes)
  and 0.96-1.06 for γ. For binary outcomes with threshold g, where the bootstrap
  of the main run over-covers β (SE/SD 1.15, coverage 0.971), the sandwich covers
  0.946.
- **Package stopping rule: valid for linear and sigmoid g, too wide for
  threshold g.** β SE/SD rises to 1.42 (continuous) and 1.19 (binary) for
  threshold g, where these fits stop furthest from a minimum (median gradient
  norm 9e-4 to 4e-3, against 1e-5 to 2e-4 converged).
- **AdamW: invalid.** β covers 0.64-0.76 and γ 0.78-0.85, with SE/SD about 0.5-0.7
  for β. Early stopping halts AdamW far from any minimum: the median gradient
  norm is 0.04-0.31 and the Hessian is indefinite in every fit (median smallest
  eigenvalue -0.03 to -0.27). The sandwich describes variation around a minimum,
  which these fits never reach. The bootstrap of the same estimators (main run)
  covers 0.94-0.98.
- **AdamW + exact γ: valid for γ only.** γ, fitted exactly with the network
  fixed, covers 0.940-0.965 with SE/SD near 1 (except binary sigmoid, where the
  indefinite Hessian inflates the SEs); β, left where AdamW stopped, covers
  0.64-0.76 as before. So within one fit the sandwich works for the parameters at
  a minimum and fails for those that are not.

This explains the July attempts (`AUDIT_2026-07-29.md`, sections 3f-3g), whose
sandwich SEs under-covered (β SE/SD 0.74-0.97): they were computed at
minibatch-Adam fits and from the score of (β, γ) with ĝ held fixed, without the
network's estimation.

The Hessian at converged fits is slightly indefinite in about half of them
(median smallest eigenvalue -4e-6, at worst -3.5e-3, against the penalty's
curvature of 1e-3): flat directions among the network weights, which do not
affect β and γ.

**Cost.** One sandwich takes about 15 s on one thread (under 2 s on 10), against
300-1,000 CPU-seconds for a 1,000-refit bootstrap. A converged fit takes about
2-7 s on one thread, against 0.5-2 s with the package's stopping rule.

## SplinePLSI

`spline_sandwich.py` does the same for SplinePLSI with its knots held at their
fitted values, as for a regression spline: about 18 parameters (β in the tangent
plane, γ and 7 spline coefficients), with the engine's objective (RSS +
α|w|² for continuous outcomes, negative log-likelihood + α|w|²/2 for binary
ones); its basis and objective match the engine's exactly. The fit's knots sit
at quantiles of x'β and move with β, so with them fixed the fit is not quite a
stationary point (median gradient norm 4.8, on the objective's sum-over-rows
scale); a few Newton steps on the fixed-knot objective make it one (gradient
5e-13), move β by about 0.001 and leave the SEs and coverage as they are. The
Hessian is positive definite in every fit.

500 datasets per setting with the seeds and fit seeds of the main simulation;
SE/SD / coverage as fitted, with the main run's SplinePLSI bootstrap (B = 999 or
1,000) on the same datasets:

| setting | β sandwich | γ sandwich | β bootstrap | γ bootstrap |
|---|---|---|---|---|
| continuous, linear | 1.01 / 0.954 | 0.98 / 0.948 | 1.03 / 0.957 | 0.98 / 0.942 |
| continuous, sigmoid | 1.01 / 0.949 | 0.97 / 0.944 | 1.02 / 0.956 | 0.98 / 0.943 |
| continuous, threshold | 0.98 / 0.941 | 0.97 / 0.944 | 1.02 / 0.959 | 0.98 / 0.945 |
| binary, linear | 1.04 / 0.947 | 1.00 / 0.949 | 1.14 / 0.969 | 1.01 / 0.939 |
| binary, sigmoid | 1.01 / 0.948 | 1.01 / 0.951 | 1.02 / 0.948 | 1.01 / 0.931 |
| binary, threshold | 1.06 / 0.954 | 1.01 / 0.957 | 1.34 / 0.977 | 1.05 / 0.958 |

The sandwich is valid in every setting: β covers 0.941-0.954 and γ
0.944-0.957, with SE/SD 0.98-1.06. Where the spline's bootstrap is too wide
(binary outcomes, linear and threshold g, SE/SD 1.14 and 1.34), the sandwich is
not (1.04 and 1.06). The bootstrap column is from the main run's fits, computed
on BigPurple; SplinePLSI works in double precision, but those fits were not
checked to be identical to these.

## Sandwich and bootstrap on the same fits

`study.py` (BigPurple, job 28777130): for each of 200 datasets per setting, one
NeuralPLSI fit, its sandwich SEs and a B = 500 cold-start bootstrap of the same
fit, with the converged and the package stopping rules. SE/SD uses the median
SE over the datasets; coverage is the sandwich's Wald interval and the
bootstrap's percentile interval; the per-fit ratio is the median of sandwich SE
/ bootstrap SE (`matched_comparison.csv`). Converged fits:

| setting | β SE/SD, sandwich / bootstrap | β coverage, sandwich / bootstrap | β per-fit SE ratio | γ coverage, sandwich / bootstrap |
|---|---|---|---|---|
| continuous, linear | 1.01 / 1.03 | 0.955 / 0.956 | 0.99 | 0.935 / 0.938 |
| continuous, sigmoid | 1.02 / 1.03 | 0.958 / 0.957 | 0.99 | 0.938 / 0.933 |
| continuous, threshold | 1.00 / 1.08 | 0.949 / 0.963 | 0.92 | 0.943 / 0.938 |
| binary, linear | 0.97 / 0.97 | 0.934 / 0.934 | 1.00 | 0.947 / 0.935 |
| binary, sigmoid | 0.97 / 0.98 | 0.936 / 0.934 | 0.99 | 0.948 / 0.935 |
| binary, threshold | 0.95 / 1.12 | 0.937 / 0.969 | 0.87 | 0.953 / 0.947 |

- **For linear and sigmoid g the two agree fit by fit**: the SEs differ by 1% at
  the median, and coverage by at most 0.002 for β.
- **For threshold g the sandwich is 8-13% narrower**, and the bootstrap's
  over-coverage (0.963 and 0.969) goes: the sandwich covers 0.949 and 0.937. So
  the bootstrap's excess width there is the bootstrap's, not the estimator's.
- With the package's stopping rule, coverage is within 0.006 and SE/SD within
  0.03 of these (in `matched_comparison.csv`); on these fits the sandwich holds for threshold g
  as well when the median SE is used. The mean SE, used in the tables above, is
  pulled up by 1-4% of fits whose Hessian is nearly singular in a direction that
  involves β; their SEs are very large.
- **Cost**: a converged fit took 3.9 s, its sandwich 1.5 s and its B = 500
  bootstrap 64 s on 40 threads (about 2,500 CPU-seconds); with the package's rule,
  0.5 s, 1.0 s and 8.8 s.

## Adam to first-order optimality, full-batch and minibatch

`adam_first_order.py` runs Adam on the penalized objective itself (not AdamW's
decoupled weight decay, whose fixed points do not solve the objective's
first-order condition), on all rows, without early stopping, with a decaying
learning rate, from the engine's initialization, until the gradient norm is
below 1e-4; `adam_monte_carlo.py` adds the sandwich, 200 datasets per setting.
Full batch: learning rate 3e-3 / (1 + step/5000). Minibatch: 64 rows, 1e-3 / (1 +
step/20000), 160,000 steps (about the full-batch fits' row-gradients). Median
SE / SD and coverage:

| setting | full batch: β | γ | minibatch: β | γ | minibatch final gradient |
|---|---|---|---|---|---|
| continuous, linear | 1.03 / 0.953 | 0.96 / 0.942 | 1.02 / 0.955 | 0.96 / 0.942 | 1.4e-2 |
| continuous, sigmoid | 1.07 / 0.957 | 0.97 / 0.938 | 1.04 / 0.958 | 0.96 / 0.937 | 2.1e-2 |
| continuous, threshold | 1.04 / 0.954 | 0.98 / 0.942 | 1.04 / 0.953 | 0.96 / 0.938 | 3.6e-2 |
| binary, linear | 1.01 / 0.956 | 1.05 / 0.958 | 1.01 / 0.956 | 1.04 / 0.965 | 8.9e-4 |
| binary, sigmoid | 1.06 / 0.956 | 1.06 / 0.965 | 1.03 / 0.955 | 1.06 / 0.963 | 3.9e-3 |
| binary, threshold | 1.00 / 0.944 | 1.12 / 0.962 | 1.04 / 0.954 | 1.02 / 0.965 | 2.8e-3 |

- **Full batch reaches first-order optimality** (99.8% of fits, median 4,500
  steps) and its sandwich is valid everywhere, as for converged L-BFGS: what
  matters is the stationary point, not the optimizer's order. It is slow:
  about 80 s per fit with 10 fits sharing a laptop, against seconds for L-BFGS.
- **Minibatch does not reach it** (1.4% of fits; median gradient 7.5e-3; objective
  0.0001-0.004 above full batch), **yet its sandwich is valid too.** The
  remaining gradient lies mostly in flat directions among the network weights,
  which barely move β and γ. What decides validity is how far β̂ and γ̂ are from
  the stationary point (about the β and γ part of the Newton step H⁻¹∇Q, in SE
  units), not the gradient norm over all 1,164 weights.
- So what breaks the sandwich for typical training (the AdamW baseline above,
  β coverage 0.64-0.76) is how it ends, not minibatches as such: early stopping
  on a 20% validation split, about 10 times larger final gradients (0.04-0.31),
  80% of the rows and decoupled weight decay. Which of these matters most is not
  separated here.

## Not yet done

- Cox outcomes (score residuals).
- The Newton-step diagnostic (β and γ part of H⁻¹∇Q in SE units) as the stated
  validity check, in place of the gradient norm.

## Files and how to reproduce

- `sandwich_monte_carlo.csv`: SE/SD and coverage per arm, setting and block, with
  their range over coordinates, the median gradient norm and smallest Hessian
  eigenvalue, the share of fits with an indefinite Hessian, and median times.
- `matched_comparison.csv`: the sandwich and the bootstrap on the same fits.
- Raw rows (one per fit) are in `output/sandwich/`, not versioned.

```bash
sh reproduce/sandwich/build_converged.sh /tmp/libneuralplsi_converged.so
NEURALPLSI_LIBRARY=/tmp/libneuralplsi_converged.so python reproduce/sandwich/monte_carlo.py output/sandwich/converged.jsonl 200
python reproduce/sandwich/monte_carlo.py output/sandwich/current.jsonl 200
python reproduce/sandwich/monte_carlo.py output/sandwich/adam.jsonl 200 adam
python reproduce/sandwich/monte_carlo.py output/sandwich/adamgamma.jsonl 200 adam_exact_gamma
python reproduce/sandwich/spline_monte_carlo.py output/sandwich/spline.jsonl 500
python reproduce/sandwich/adam_monte_carlo.py output/sandwich/adam_full_batch.jsonl 200
python reproduce/sandwich/adam_monte_carlo.py output/sandwich/adam_minibatch.jsonl 200 minibatch
python reproduce/sandwich/summarize.py
python reproduce/sandwich/summarize_matched.py   # after copying the matched rows to output/sandwich/matched
```

Each Monte Carlo run uses 10 processes and takes 30-40 minutes on a laptop. The
matched study: `sbatch reproduce/sandwich/study.slurm` from the repository root
on BigPurple.
