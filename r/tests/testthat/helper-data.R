# Small simulated data on a 1/1024 grid, so every value is exactly
# representable in float32 and other languages read the same numbers.
simulate <- function(family = "continuous", n = 150, q = 2, seed = 1) {
  set.seed(seed)
  grid <- function(v) round(v * 1024) / 1024
  x <- matrix(grid(rnorm(n * 3)), n, 3, dimnames = list(NULL, c("lead", "cadmium", "mercury")))
  z <- matrix(grid(rnorm(n * q)), n, q, dimnames = list(NULL, c("age", "sex")[seq_len(q)]))
  eta <- sin(1.5 * drop(x %*% c(0.6, -0.6, 0.5))) + drop(z %*% c(0.5, -0.3)[seq_len(q)])
  y <- switch(family,
    continuous = grid(eta + rnorm(n, sd = 0.2)),
    binary = rbinom(n, 1, stats::plogis(eta)),
    cox = cbind(time = grid(rexp(n, exp(eta))) + 1 / 1024, status = rbinom(n, 1, 0.8)))
  list(x = x, z = z, y = y)
}

# Small networks keep the tests fast; settings are otherwise the defaults.
small_fit <- function(data, family, hidden_units = 8, max_iter = 300, ...) {
  neuralplsi(data$x, data$z, data$y, family = family, hidden_units = hidden_units,
             max_iter = max_iter, ...)
}

all_families <- c("continuous", "binary", "cox")

# The repository root when the tests run from a NeuralPLSI checkout (r/tests/testthat),
# NULL when they run from an installed or checked tarball.
checkout <- function() {
  root <- normalizePath(test_path("..", "..", ".."), mustWork = FALSE)
  found <- file.exists(file.path(root, c("core/include/neuralplsi.h", "r/src/neuralplsi.h",
                                         "python/neuralplsi/_version.py")))
  if (all(found)) root
}
