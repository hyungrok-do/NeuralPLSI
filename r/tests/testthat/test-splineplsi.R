test_that("SplinePLSI fits and predicts for every family", {
  for (family in all_families) {
    data <- simulate(family)
    fit <- splineplsi(data$x, data$z, data$y, family = family)
    expect_s3_class(fit, c("splineplsi", "plsi"))
    expect_identical(names(coef(fit)), c("lead", "cadmium", "mercury", "age", "sex"))
    expect_equal(sum(fit$beta^2), 1, tolerance = 1e-9)
    expect_gte(fit$beta[[1]], 0)
    expect_length(fit$spline, 7)
    expect_length(fit$knots, 11)
    expect_true(fit$diagnostics$converged)
    link <- predict(fit)
    index <- predict(fit, type = "index")
    expect_equal(link - index, drop(data$z %*% fit$gamma), tolerance = 1e-12)
    expect_identical(index, g_function(fit, data$x %*% fit$beta))
    expected <- switch(family, continuous = link, binary = stats::plogis(link), cox = exp(link))
    expect_identical(predict(fit, type = "response"), expected)
    expect_identical(predict(fit, type = "risk"), exp(link))
    if (family == "cox") expect_equal(g_function(fit, 0), 0, tolerance = 1e-12)
  }
})

test_that("SplinePLSI settings, seeds and missing covariates", {
  data <- simulate(q = 0)
  fit <- splineplsi(data$x, NULL, data$y, num_knots = 4, degree = 2, seed = 5)
  expect_identical(fit$settings, list(num_knots = 4L, degree = 2L, alpha = 1e-6, max_iter = 200L,
                                      tol = 1e-6))
  expect_length(fit$spline, 5)
  expect_identical(coef(splineplsi(data$x, NULL, data$y, num_knots = 4, degree = 2, seed = 5)),
                   coef(fit))
  expect_identical(predict(fit, data$x), predict(fit))
})

test_that("the spline is constant outside the boundary knots", {
  data <- simulate()
  fit <- splineplsi(data$x, data$z, data$y)
  ends <- range(fit$knots)
  expect_identical(g_function(fit, ends[1] - c(1, 10)), rep(g_function(fit, ends[1]), 2))
  expect_identical(g_function(fit, ends[2] + c(1, 10)), rep(g_function(fit, ends[2]), 2))
})
