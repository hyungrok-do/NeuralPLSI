test_that("model settings are validated", {
  d <- simulate(n = 30)
  fit <- function(...) neuralplsi(d$x, d$z, d$y, max_iter = 1, ...)
  expect_error(fit(family = "poisson"), "should be one of")
  expect_error(neuralplsi(d$x, d$z, cbind(1:30, 1), family = "cox", add_intercept = TRUE),
               "cannot identify an intercept")
  expect_error(fit(hidden_units = 0), "hidden_units must be an integer >= 1")
  expect_error(fit(hidden_units = 2.5), "hidden_units must be an integer >= 1")
  expect_error(fit(n_hidden_layers = TRUE), "n_hidden_layers must be an integer >= 1")
  expect_error(neuralplsi(d$x, d$z, d$y, max_iter = -1), "max_iter must be an integer >= 0")
  expect_error(fit(batch_size = NA), "batch_size must be an integer >= 1")
  expect_error(fit(weight_decay = -1), "weight_decay must be finite and nonnegative")
  expect_error(fit(learning_rate = 0), "learning_rate must be finite and positive")
  expect_error(fit(add_intercept = "yes"), "add_intercept must be TRUE or FALSE")
  expect_error(fit(activation = "sigmoid"), "should be one of")
  expect_error(fit(optimizer = "sgd"), "should be one of")
  expect_error(fit(seed = -1), "seed must be an integer in \\[0, 2\\^53\\]")
  expect_error(fit(seed = 0.5), "seed must be an integer")
  expect_error(fit(seed = 2^54), "seed must be an integer")
  expect_error(fit(hidden_units = 20000), "Invalid native network configuration")
  expect_error(splineplsi(d$x, d$z, d$y, num_knots = 1), "num_knots must be an integer >= 2")
  expect_error(splineplsi(d$x, d$z, d$y, degree = -1), "degree must be an integer >= 0")
  expect_error(splineplsi(d$x, d$z, d$y, max_iter = 0), "max_iter must be an integer >= 1")
  expect_error(splineplsi(d$x, d$z, d$y, alpha = -1), "alpha must be finite and nonnegative")
})

test_that("data are validated as in Python", {
  d <- simulate(n = 30)
  fit <- function(x = d$x, z = d$z, y = d$y, family = "continuous")
    neuralplsi(x, z, y, family = family, max_iter = 1)
  for (model in list(fit, function(x = d$x, z = d$z, y = d$y, family = "continuous")
                       splineplsi(x, z, y, family = family))) {
    expect_error(model(x = d$x[, 0]), "at least one exposure")
    expect_error(model(x = d$x[1:29, ]), "same number of rows")
    expect_error(model(x = d$x[1, , drop = FALSE], z = d$z[1, , drop = FALSE], y = 1),
                 "same number of rows")
    x <- d$x
    x[3, 2] <- NA
    expect_error(model(x = x), "only finite values")
    expect_error(model(y = replace(d$y, 1, Inf)), "only finite values")
    expect_error(model(x = data.frame(a = letters[1:30])), "only numeric columns")
    expect_error(model(x = matrix("a", 30, 2)), "numeric matrix or data frame")
    expect_error(model(y = cbind(d$y, d$y)), "a vector or a single column")
    expect_error(model(family = "binary"), "Binary outcomes must be coded 0 or 1")
    expect_error(model(family = "cox"), "two-column matrix")
    expect_error(model(y = cbind(-1, rep(1, 30)), family = "cox"), "nonnegative durations")
    expect_error(model(y = cbind(1, rep(2, 30)), family = "cox"), "binary event indicators")
    expect_error(model(y = cbind(1, rep(0, 30)), family = "cox"), "at least one event")
    expect_error(model(z = cbind(lead = d$z[, 1])), "distinct names")
  }
  expect_error(fit(x = d$x * 1e39), "x has values outside the float32 range")
  skip_if_not_installed("survival")
  expect_error(fit(y = survival::Surv(d$y), family = "continuous"), "family = \"cox\"")
  expect_error(fit(y = survival::Surv(rep(1, 30), rep(2, 30), rep(1, 30)), family = "cox"),
               "right-censored")
})

test_that("prediction and bootstrap arguments are validated", {
  d <- simulate(n = 30)
  fit <- neuralplsi(d$x, d$z, d$y, max_iter = 1)
  expect_error(predict(fit, d$x[, 1:2], d$z), "newx must have 3 columns")
  expect_error(predict(fit, d$x), "newz is required")
  expect_error(predict(fit, d$x, d$z[1:5, ]), "same number of rows")
  expect_error(predict(fit, replace(d$x, 1, NaN), d$z), "only finite values")
  expect_error(predict(fit, type = "class"), "should be one of")
  expect_error(g_function(fit, c(0, NA)), "numeric and finite")
  expect_error(g_function(list(), 0), "neuralplsi or splineplsi fit")
  expect_error(plsi_bootstrap(fit, B = 1), "B must be an integer >= 2")
  expect_error(plsi_bootstrap(fit, level = 1), "strictly between zero and one")
  expect_error(plsi_bootstrap(fit, n_jobs = 0), "nonzero 32-bit integer")
  expect_error(plsi_bootstrap(fit, quantile = "nearest"), "should be one of")
  expect_error(plsi_bootstrap(fit, g_grid = c(0, Inf)), "nonempty and finite")
  expect_error(plsi_bootstrap(fit, d$x[, 1:2], d$z, d$y), "dimensions differ")
  expect_error(confint(plsi_bootstrap(fit, B = 3, n_jobs = 1), level = 2), "strictly between")
})

test_that("Cox resamples without events fail the whole bootstrap", {
  d <- simulate(n = 30)
  y <- cbind(time = seq_len(30), status = c(1, rep(0, 29)))
  fit <- neuralplsi(d$x, d$z, y, family = "cox", max_iter = 1)
  expect_error(plsi_bootstrap(fit, B = 20, n_jobs = 2), "Cox data need at least one event")
})

test_that("float32 conversion rounds to nearest", {
  # Ties go to even, also at the top of the range, where they overflow.
  values <- c(1 / 3, 0.1, 2^-149, 2^24 + 1, 2^24 + 3, -2^-151, (2 - 2^-23 + 2^-25) * 2^127,
              (2 - 2^-24) * 2^127)
  expect_identical(.Call(neuralplsi:::C_as_float32, values),
                   c(11184811 / 2^25, 13421773 / 2^27, 2^-149, 2^24, 2^24 + 4, -0,
                     (2 - 2^-23) * 2^127, Inf))
})
