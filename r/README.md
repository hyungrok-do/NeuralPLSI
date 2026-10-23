# neuralplsi (R)

Neural-network partial-linear single-index models, `g(x'beta) + z'gamma`, for
continuous, binary and right-censored outcomes, with bootstrap inference. The
package wraps the C++ engine in `core/`, the same one the Python package uses:
on the same machine, R and Python give identical fits, bootstrap draws and
predictions, and read each other's model files.

`neuralplsi()` fits the neural-network link; `splineplsi()` fits the B-spline
comparator of Wang et al. (2020). The defaults are those of the Python classes.

## Installation

The package needs a C++17 compiler and no other R packages (`survival` is
optional, for `Surv` outcomes). From GitHub:

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
multiply-adds, and float32 results then differ from Python's.

## Fitting

`x` holds the exposures and `z` the covariates (`NULL` for none), as numeric
matrices or data frames. Their column names label beta and gamma.

```r
library(neuralplsi)
set.seed(1)
n <- 500
x <- matrix(rnorm(3 * n), n, 3, dimnames = list(NULL, c("lead", "cadmium", "mercury")))
z <- cbind(age = rnorm(n), sex = rbinom(n, 1, 0.5))
index <- drop(x %*% c(0.6, -0.6, 0.5))

y <- sin(2 * index) + 0.5 * z[, "age"] - 0.3 * z[, "sex"] + rnorm(n, sd = 0.3)
fit <- neuralplsi(x, z, y)
fit
predict(fit, x[1:5, ], z[1:5, ])

yb <- rbinom(n, 1, plogis(sin(2 * index) + 0.5 * z[, "age"]))
fitb <- neuralplsi(x, z, yb, family = "binary", add_intercept = TRUE)
predict(fitb, x[1:5, ], z[1:5, ], type = "response")   # probabilities
```

Cox outcomes take a right-censored `survival::Surv` object or a two-column
matrix of times and event indicators:

```r
library(survival)
time <- rexp(n, exp(sin(2 * index) + 0.5 * z[, "age"]))
status <- rbinom(n, 1, 0.8)
fitc <- neuralplsi(x, z, Surv(time, status), family = "cox")
predict(fitc, x[1:5, ], z[1:5, ], type = "risk")        # exp of the linear predictor

spline <- splineplsi(x, z, Surv(time, status), family = "cox")
coef(spline)
```

`predict()` types are `"link"` (the linear predictor), `"response"` (identity,
probability, or relative risk by family), `"risk"` (the exponentiated linear
predictor) and `"index"` (`g(x'beta)` alone). `g_function(fit, t)` evaluates
the link at index values `t`.

## Bootstrap

`plsi_bootstrap()` refits the model to `B` pairs resamples (1000 by default) in
C++ on native threads and returns the fit with the draws attached, so
`summary()`, `confint()` and `plot()` show bootstrap standard errors, percentile
intervals and a pointwise band for `g`:

```r
fit <- plsi_bootstrap(fit, B = 1000, seed = 0, n_jobs = -1)
summary(fit)
confint(fit, level = 0.9)
plot(fit)
fit$bootstrap$draws[1:3, ]
```

Replicate `b` resamples rows with seed `seed + b` and refits from a fresh
initialization with seed `seed + 1337 + b` (protocol
`splitmix64_bootstrap_v1`). The draws do not depend on `n_jobs` or `B`, and
NeuralPLSI and SplinePLSI see the same resamples for the same seed. `n_jobs =
-1` uses all available CPUs, capped by `SLURM_CPUS_PER_TASK` and cgroup quotas.

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
`python/neuralplsi`, compare fits, bootstrap draws and model files with Python
exactly:

```bash
R CMD INSTALL r
cd r/tests/testthat && PYTHON=python3 Rscript -e 'testthat::test_dir(".", package = "neuralplsi", load_package = "installed")'
```

`R CMD check` runs the same tests from the tarball and skips those two.

## References

Do, H., Wang, Y., Liu, M. and Lee, M. (2025). Neural network-based
partial-linear single-index models for environmental mixtures analysis.
[arXiv:2512.11593](https://arxiv.org/abs/2512.11593).

Wang, Y., Wu, Y., Jacobson, M. H. et al. (2020). A family of partial-linear
single-index models for analyzing complex environmental exposures with
continuous, categorical, time-to-event, and longitudinal health outcomes.
*Environmental Health* 19, 96.
