test_that("NeuralPLSI fits and predicts for every family", {
  for (family in all_families) {
    data <- simulate(family)
    fit <- small_fit(data, family)
    expect_s3_class(fit, c("neuralplsi", "plsi"))
    expect_identical(names(coef(fit)), c("lead", "cadmium", "mercury", "age", "sex"))
    expect_equal(sqrt(sum(fit$beta^2)), 1, tolerance = 1e-5)
    expect_gte(fit$beta[[1]], 0)
    expect_identical(fit$family, family)
    link <- predict(fit)
    expect_length(link, 150)
    expect_identical(predict(fit, data$x, data$z), link)
    expect_equal(predict(fit, type = "risk"), exp(link), tolerance = 1e-6)
    index <- predict(fit, data$x, type = "index")
    expect_equal(index, g_function(fit, data$x %*% fit$beta), tolerance = 1e-5)
    expect_equal(link - index, drop(data$z %*% fit$gamma), tolerance = 1e-5)
    response <- predict(fit, type = "response")
    expected <- switch(family, continuous = link, binary = stats::plogis(link), cox = exp(link))
    expect_equal(response, expected, tolerance = 1e-6)
  }
})

test_that("a fit is a deterministic function of the seed", {
  data <- simulate()
  a <- small_fit(data, "continuous", seed = 3)
  expect_identical(coef(small_fit(data, "continuous", seed = 3)), coef(a))
  expect_false(identical(coef(small_fit(data, "continuous", seed = 4)), coef(a)))
  expect_identical(a$seed, 3)
})

test_that("defaults match the Python NeuralPLSI class", {
  data <- simulate(n = 60)
  fit <- neuralplsi(data$x, data$z, data$y, max_iter = 0)
  expect_identical(fit$settings, list(hidden_units = 32L, n_hidden_layers = 2L,
    activation = "LeakyReLU", weight_decay = 0.0010000000474974513, max_iter = 0L,
    add_intercept = FALSE, initial = FALSE, optimizer = "lbfgs", batch_size = 64L,
    learning_rate = 0.0010000000474974513, patience = 10L))
})

test_that("intercepts, data frames, vectors and the AdamW baseline work", {
  data <- simulate("binary")
  fit <- small_fit(data, "binary", add_intercept = TRUE)
  expect_identical(names(coef(fit))[6], "(Intercept)")
  expect_identical(fit$intercept, coef(fit)[["(Intercept)"]])
  frame <- small_fit(list(x = as.data.frame(data$x), z = as.data.frame(data$z),
                          y = data$y == 1), "binary", add_intercept = TRUE)
  expect_identical(coef(frame), coef(fit))
  single <- neuralplsi(data$x[, 1], NULL, data$y, family = "binary", hidden_units = 4, max_iter = 50)
  expect_identical(names(coef(single)), "x1")
  expect_equal(unname(coef(single)), 1, tolerance = 1e-5)
  adam <- small_fit(data, "binary", optimizer = "adam", max_iter = 20, batch_size = 32)
  expect_identical(adam$settings$optimizer, "adam")
  expect_lte(adam$diagnostics$best_epoch, 20)
})

test_that("models without covariates use z = NULL", {
  data <- simulate(q = 0)
  expect_identical(ncol(data$z), 0L)
  fit <- neuralplsi(data$x, NULL, data$y, hidden_units = 8, max_iter = 300)
  expect_identical(fit$gamma, stats::setNames(numeric(0), character(0)))
  expect_identical(coef(neuralplsi(data$x, data$z, data$y, hidden_units = 8, max_iter = 300)),
                   coef(fit))
  expect_identical(predict(fit, data$x), predict(fit))
  expect_identical(predict(fit, data$x), predict(fit, data$x, type = "index"))
})

test_that("Cox outcomes accept Surv objects and two-column matrices", {
  skip_if_not_installed("survival")
  data <- simulate("cox")
  fit <- small_fit(data, "cox")
  surv <- small_fit(list(x = data$x, z = data$z,
                         y = survival::Surv(data$y[, 1], data$y[, 2])), "cox")
  expect_identical(coef(surv), coef(fit))
  expect_null(fit$intercept)
})

test_that("prediction matches columns by name, or by position without fitted names", {
  data <- simulate()
  fit <- small_fit(data, "continuous")
  shuffled <- predict(fit, data$x[, 3:1], data$z[, 2:1])
  expect_identical(shuffled, predict(fit))
  expect_identical(predict(fit, as.data.frame(data$x[, 3:1]), as.data.frame(data$z)), predict(fit))
  expect_identical(predict(fit, unname(data$x), unname(data$z)), predict(fit))
  expect_identical(predict(fit, data$x[1:5, ], data$z[1:5, ]), predict(fit)[1:5])
  other <- data$x
  colnames(other) <- c("a", "b", "c")
  expect_identical(predict(fit, other, data$z), predict(fit))
  colnames(other) <- c("mercury", "lead", "arsenic")
  expect_error(predict(fit, other, data$z),
               "newx has some of the fitted column names but not cadmium")
  expect_error(predict(fit, data$x, cbind(age = data$z[, 1], weight = 0)),
               "newz has some of the fitted column names but not sex")
  expect_error(predict(fit, cbind(data$x, arsenic = 0), data$z), "newx must have 3 columns")
})

test_that("an (Intercept) column is refused only next to a fitted intercept", {
  data <- simulate("binary")
  z <- stats::model.matrix(~ age + sex, as.data.frame(data$z))
  expect_identical(colnames(z), c("(Intercept)", "age", "sex"))
  expect_error(small_fit(list(x = data$x, z = z, y = data$y), "binary", add_intercept = TRUE),
               "column named \\(Intercept\\).*model.matrix\\(...\\)\\[, -1\\]")
  x <- cbind(data$x, `(Intercept)` = 1)
  expect_error(small_fit(list(x = x, z = data$z, y = data$y), "binary", add_intercept = TRUE),
               "column named \\(Intercept\\)")
  fit <- small_fit(list(x = data$x, z = z, y = data$y), "binary")
  expect_identical(names(coef(fit)), c("lead", "cadmium", "mercury", "(Intercept)", "age", "sex"))
  fit <- small_fit(list(x = data$x, z = z[, -1], y = data$y), "binary", add_intercept = TRUE)
  expect_identical(names(coef(fit)), c("lead", "cadmium", "mercury", "age", "sex", "(Intercept)"))
  spline <- splineplsi(data$x, z, data$y, family = "binary")
  expect_identical(names(coef(spline)), c("lead", "cadmium", "mercury", "(Intercept)", "age", "sex"))
})

test_that("plot draws g and returns the curve", {
  data <- simulate()
  fit <- small_fit(data, "continuous")
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  curve <- plot(fit)
  expect_named(curve, c("index", "g"))
  expect_identical(curve$g, g_function(fit, curve$index))
  expect_equal(range(curve$index), range(data$x %*% fit$beta))
})
