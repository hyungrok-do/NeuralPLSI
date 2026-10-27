# neuralplsi (R)

Neural-network partial-linear single-index models, `g(x'beta) + z'gamma`, for
continuous, binary and right-censored outcomes, with bootstrap or sandwich inference. The
package wraps the C++ engine in `core/`, the same one the Python package uses:
on the same machine, R and Python give identical fits, bootstrap draws and
NeuralPLSI predictions, and read each other's model files. SplinePLSI
predictions and `g_function()` values agree to rounding error (a few times
1e-15 on the test data), not bit for bit, because R and numpy sum the final
matrix products in different orders.

`neuralplsi()` fits the neural-network link; `splineplsi()` fits the B-spline
comparator of Wang et al. (2020). Their model settings default to those of the
Python classes, and `seed = 0` matches Python's default `random_state = 0` for
`NeuralPLSI.fit()`. The differences from Python are:

- `plsi_bootstrap()` draws `B = 1000` resamples by default; Python's
  `inference_bootstrap()` draws `n_samples = 200`.
- `g_grid = NULL` bootstraps `g` at 101 equally spaced points over the range of
  the fitted index; in Python, `g_grid=None` bootstraps no `g` values.
- Seeds lie in [0, 2^53], the integers an R double holds exactly; Python
  accepts [0, 2^64).
- `splineplsi()` uses `seed = 0` by default, where Python's
  `SplinePLSI.fit(random_state=None)` draws a random seed.
- `quantile = "weibull"` is available for SplinePLSI too; Python's
  `SplinePLSI.inference_bootstrap()` uses linear quantiles only.

## Installation

The package needs a C++17 compiler with `std::filesystem` (on macOS, 10.15 or
later) and no other R packages (`survival` is optional, for `Surv` outcomes).
From GitHub:

```r
remotes::install_github("hyungrok-do/NeuralPLSI", subdir = "r")
```

From a checkout of the repository:

```bash
R CMD INSTALL r
```

or as a tarball:

```bash
R CMD build r
R CMD INSTALL neuralplsi_0.1.0.tar.gz
```

`configure` compiles the engine with `-ffp-contract=off` when the compiler
accepts it, as the CMake and Python builds do. Without it, compilers may fuse
multiply-adds, and float32 results then differ from Python's. It also links
`-lstdc++fs` when `std::filesystem` needs it, as with GCC 8.

## Fitting

`x` holds the exposures and `z` the covariates (`NULL` for none), as numeric
matrices or data frames. Their column names label beta and gamma.

The example follows the design of the paper's simulations: six exposures with
pairwise correlations of 0.3, unit-length index weights `beta` (`bpa` has no
effect), the sigmoid link `g(u) = 5 (plogis(2u) - 1/2)`, and age and sex as
covariates.

```r
library(neuralplsi)
set.seed(1)
n <- 1000
x <- matrix(rnorm(6 * n), n, 6) * sqrt(0.7) + rnorm(n) * sqrt(0.3)
colnames(x) <- c("lead", "cadmium", "mercury", "arsenic", "pfoa", "bpa")
z <- cbind(age = rnorm(n), sex = rbinom(n, 1, 0.5))
beta <- c(0.5, 0.5, -0.5, 0.4, 0.3, 0)
g <- function(u) 5 * (plogis(2 * u) - 0.5)
eta <- g(drop(x %*% beta)) + 0.5 * z[, "age"] - 0.3 * z[, "sex"]

y <- eta + rnorm(n)
fit <- neuralplsi(x, z, y)
fit
predict(fit, x[1:5, ], z[1:5, ])

yb <- rbinom(n, 1, plogis(eta))
fitb <- neuralplsi(x, z, yb, family = "binary", add_intercept = TRUE)
predict(fitb, x[1:5, ], z[1:5, ], type = "response")   # probabilities
```

Cox outcomes take a right-censored `survival::Surv` object or a two-column
matrix of times and event indicators:

```r
library(survival)
event_time <- rexp(n, exp(eta))
censor_time <- rexp(n, 0.2)
time <- pmin(event_time, censor_time)
status <- as.integer(event_time <= censor_time)         # 72% events
fitc <- neuralplsi(x, z, Surv(time, status), family = "cox")
predict(fitc, x[1:5, ], z[1:5, ], type = "risk")        # exp of the linear predictor

spline <- splineplsi(x, z, Surv(time, status), family = "cox")
coef(spline)
```

`predict()` types are `"link"` (the linear predictor), `"response"` (identity,
probability, or relative risk by family), `"risk"` (the exponentiated linear
predictor) and `"index"` (`g(x'beta)` alone). `g_function(fit, t)` evaluates
the link at index values `t`.

New data for `predict()` and `plsi_bootstrap()` need as many columns as the
fit. Columns are matched by name when the data have any of the fitted column
names, and must then have all of them; by position when they have none of them
or no column names.

## Bootstrap

`plsi_bootstrap()` refits the model to `B` pairs resamples (1000 by default) in
C++ on native threads and returns the fit with the draws attached, so
`summary()`, `confint()` and `plot()` show bootstrap standard errors, percentile
intervals and a pointwise band for `g`:

```r
fit <- plsi_bootstrap(fit, B = 1000, seed = 0, n_jobs = -1)
summary(fit)
#> NeuralPLSI, continuous outcome, n = 1000
#> Call: neuralplsi(x = x, z = z, y = y)
#>
#> Index coefficients (beta, unit length):
#>           Estimate Std. Error    2.5 %   97.5 %
#> lead     0.4704566    0.02320  0.41713  0.51040
#> cadmium  0.5373262    0.01954  0.49819  0.57449
#> mercury -0.4974372    0.01447 -0.52867 -0.47349
#> arsenic  0.4099630    0.01963  0.37360  0.44964
#> pfoa     0.2728288    0.02055  0.22926  0.30898
#> bpa     -0.0004116    0.02016 -0.03653  0.04016
#>
#> Covariate coefficients (gamma):
#>     Estimate Std. Error   2.5 %  97.5 %
#> age   0.4690    0.03049  0.4132  0.5314
#> sex  -0.3199    0.05933 -0.4488 -0.2056
#>
#> Network: 2 hidden layers of 32 ELU units. L-BFGS: 346 iterations, penalized objective 0.918552.
#> Bootstrap: B = 1000 pairs resamples with cold-start refits, seed 0, 95% percentile intervals (linear quantiles), 10 threads.
confint(fit, level = 0.9)
plot(fit)
fit$bootstrap$draws[1:3, ]
```

Every interval contains the true value, and the one for `bpa`, which has no
effect, contains zero. The 1000 refits took about two and a half minutes on a
10-core laptop (macOS arm64); other CPUs and compilers can give slightly different
numbers.

Replicate `b` resamples rows with seed `seed + b` and refits from a fresh
initialization with seed `seed + 1337 + b` (protocol
`splitmix64_bootstrap_v1`). The draws do not depend on `n_jobs` or `B`, and
NeuralPLSI and SplinePLSI see the same resamples for the same seed. `n_jobs =
-1` uses all available CPUs, capped by `SLURM_CPUS_PER_TASK` and cgroup quotas.


## Sandwich

`plsi_sandwich()` gives standard errors and Wald intervals for NeuralPLSI fits
without refitting (for Cox outcomes, with Lin and Wei's score residuals): the variance of the
minimizer of the fitted objective, computed by the engine in double precision.
`method = "full"` covers every weight of the network (a dense Hessian, for
networks of at most 8000 weights); `method = "frozen"` covers beta, gamma and
the intercept with the network held fixed (any size, slightly narrower);
`"auto"` picks `"frozen"` for `optimizer = "adam_frozen"` and `"full"`
otherwise. `summary()` and `confint()` then use it (with `inference =
"sandwich"` when a bootstrap is also attached). The result reports the
remaining Newton step in SE units and the Hessian's eigenvalue nearest zero:
the formula describes a minimizer, and a large step (well above 0.1 SE) flags a
fit that stopped short of one; continue it with a larger `max_iter` or use the
bootstrap.

```r
fit <- plsi_sandwich(fit, n_jobs = -1)
summary(fit, inference = "sandwich")
```

## Saving and sharing models

A fit keeps the serialized engine state, so `saveRDS()` and `readRDS()` keep
it usable. `save_model()` and `load_model()` write and read the engine's
`.nplsi` files, which Python's `NeuralPLSI.save()` and `NeuralPLSI.load()` and
the C++ API also use:

```r
save_model(fit, "exposure_model.nplsi")
restored <- load_model("exposure_model.nplsi")
identical(predict(restored, x, z), predict(fit))
```

In Python, the same file loads and predicts identically:

```python
from neuralplsi import NeuralPLSI
model = NeuralPLSI.load("exposure_model.nplsi")
```

`.nplsi` files hold the prediction model only: column names, training data and
bootstrap results stay in the R object. SplinePLSI fits are saved with
`saveRDS()`.

## Development

The engine sources in `src/` are copies of `core/src/neuralplsi.cpp`,
`core/src/spline.cpp` and `core/include/neuralplsi.h`, because `R CMD build`
needs a self-contained `src/`. After changing `core/`, refresh them, and check
that they match and that the R, C++ and Python versions agree:

```bash
r/tools/sync-core.sh
r/tools/sync-core.sh --check
```

Run the tests from the checkout, where they also check the copies and, if
`PYTHON` (or `python3`) has numpy, pandas and scipy to import
`python/neuralplsi`, compare fits, bootstrap draws, NeuralPLSI predictions and
model files with Python exactly, and SplinePLSI predictions to rounding error:

```bash
R CMD INSTALL r
cd r/tests/testthat && PYTHON=$(command -v python) NEURALPLSI_REQUIRE_PYTHON=true Rscript -e 'testthat::test_dir(".", package = "neuralplsi", load_package = "installed")'
```

`NEURALPLSI_REQUIRE_PYTHON=true` makes the comparison fail, rather than skip,
when that Python cannot import the package. Without it, a skipped Python test
means that no comparison ran. `R CMD check` runs the same tests from the
tarball and skips the comparison and the check of the copies.

## References

Do, H., Wang, Y., Liu, M. and Lee, M. (2025). Neural network-based
partial-linear single-index models for environmental mixtures analysis.
[arXiv:2512.11593](https://arxiv.org/abs/2512.11593).

Wang, Y., Wu, Y., Jacobson, M. H. et al. (2020). A family of partial-linear
single-index models for analyzing complex environmental exposures with
continuous, categorical, time-to-event, and longitudinal health outcomes.
*Environmental Health* 19, 96.
