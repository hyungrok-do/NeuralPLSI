test_that("bootstraps attach draws and intervals for both models and every family", {
  for (family in all_families) {
    data <- simulate(family)
    fits <- list(small_fit(data, family), splineplsi(data$x, data$z, data$y, family = family))
    for (fit in fits) {
      boot <- plsi_bootstrap(fit, B = 6, seed = 2, n_jobs = 2)$bootstrap
      expect_identical(dim(boot$draws), c(6L, 5L))
      expect_identical(colnames(boot$draws), names(coef(fit)))
      expect_identical(dimnames(boot$summary), list(names(coef(fit)),
                                                    c("mean", "se", "lower", "upper")))
      expect_equal(boot$summary[, "mean"], colMeans(boot$draws))
      expect_equal(boot$summary[, "se"], apply(boot$draws, 2, stats::sd))
      expect_identical(dim(boot$g_draws), c(6L, 101L))
      expect_identical(boot$g_grid, seq(min(data$x %*% fit$beta), max(data$x %*% fit$beta),
                                        length.out = 101))
      expect_identical(boot$protocol, "splitmix64_bootstrap_v1")
      expect_lte(boot$workers, 2L)
      expect_identical(is.null(boot$spline_draws), inherits(fit, "neuralplsi"))
    }
  }
})

test_that("draws do not depend on the thread count or on B", {
  data <- simulate("binary")
  for (fit in list(small_fit(data, "binary", add_intercept = TRUE),
                   splineplsi(data$x, data$z, data$y, family = "binary"))) {
    one <- plsi_bootstrap(fit, B = 6, seed = 9, n_jobs = 1)$bootstrap
    two <- plsi_bootstrap(fit, B = 6, seed = 9, n_jobs = 2)$bootstrap
    expect_identical(one$workers, 1L)
    expect_identical(two$workers, 2L)
    expect_identical(two[c("draws", "summary", "g_draws", "g_summary")],
                     one[c("draws", "summary", "g_draws", "g_summary")])
    four <- plsi_bootstrap(fit, B = 4, seed = 9, n_jobs = 2)$bootstrap
    expect_identical(four$draws, one$draws[1:4, ])
    expect_false(identical(plsi_bootstrap(fit, B = 6, seed = 10, n_jobs = 2)$bootstrap$draws,
                           one$draws))
  }
})

test_that("replicate b refits rows bootstrap_indices(b) with fit seed seed + 1337 + b", {
  data <- simulate("cox")
  fit <- small_fit(data, "cox")
  spline <- splineplsi(data$x, data$z, data$y, family = "cox")
  neural_draws <- plsi_bootstrap(fit, B = 3, seed = 11, n_jobs = 2)$bootstrap$draws
  spline_draws <- plsi_bootstrap(spline, B = 3, seed = 11, n_jobs = 2)$bootstrap$draws
  for (b in 0:2) {
    rows <- .Call(neuralplsi:::C_bootstrap_indices, 150L, 11, b) + 1L
    refit <- small_fit(list(x = data$x[rows, ], z = data$z[rows, ], y = data$y[rows, ]), "cox",
                       seed = 11 + 1337 + b)
    expect_identical(coef(refit), neural_draws[b + 1, ])
    refit <- splineplsi(data$x[rows, ], data$z[rows, ], data$y[rows, ], family = "cox",
                        seed = 11 + 1337 + b)
    expect_identical(coef(refit), spline_draws[b + 1, ])
  }
})

test_that("summary, confint and plot use the bootstrap", {
  data <- simulate()
  fit <- small_fit(data, "continuous")
  expect_error(confint(fit), "run plsi_bootstrap")
  expect_identical(colnames(summary(fit)$coefficients), "Estimate")
  expect_output(print(summary(fit)), "No bootstrap")
  fit <- plsi_bootstrap(fit, B = 20, seed = 1, n_jobs = 2, level = 0.9)
  table <- summary(fit)$coefficients
  expect_identical(colnames(table), c("Estimate", "Std. Error", "5 %", "95 %"))
  expect_identical(table[, "Std. Error"], fit$bootstrap$summary[, "se"])
  expect_output(print(summary(fit)), "B = 20 .*, 2 threads\\.")
  expect_output(print(summary(plsi_bootstrap(fit, B = 3, n_jobs = 1))), "B = 3 .*, 1 thread\\.")
  expect_output(print(fit), "B = 20")
  intervals <- confint(fit)
  expect_identical(unname(intervals), unname(fit$bootstrap$summary[, c("lower", "upper")]))
  expect_identical(confint(fit, "age"), intervals["age", , drop = FALSE])
  expect_identical(confint(fit, 2), intervals[2, , drop = FALSE])
  wider <- confint(fit, level = 0.95)
  expect_identical(colnames(wider), c("2.5 %", "97.5 %"))
  expect_equal(unname(wider[, 1]), unname(apply(fit$bootstrap$draws, 2, stats::quantile, 0.025,
                                                 type = 7)), tolerance = 1e-12)
  weibull <- plsi_bootstrap(fit, B = 20, seed = 1, n_jobs = 2, quantile = "weibull")
  expect_identical(weibull$bootstrap$draws, fit$bootstrap$draws)
  expect_equal(unname(weibull$bootstrap$summary[, "upper"]),
               unname(apply(fit$bootstrap$draws, 2, stats::quantile, 0.975, type = 6)),
               tolerance = 1e-12)
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  curve <- plot(fit)
  expect_identical(curve$index, fit$bootstrap$g_grid)
  expect_identical(curve$lower, unname(fit$bootstrap$g_summary[, "lower"]))
  expect_named(plot(fit, grid = c(-1, 0, 1)), c("index", "g"))
})

test_that("explicit data are matched to the fitted columns as in predict()", {
  data <- simulate()
  for (fit in list(small_fit(data, "continuous"), splineplsi(data$x, data$z, data$y))) {
    boot <- plsi_bootstrap(fit, B = 3, n_jobs = 2)$bootstrap
    permuted <- plsi_bootstrap(fit, data$x[, 3:1], data$z[, 2:1], data$y, B = 3, n_jobs = 2)
    expect_identical(permuted$bootstrap[c("draws", "g_grid", "g_draws")],
                     boot[c("draws", "g_grid", "g_draws")])
    frame <- as.data.frame(data$x)[c("mercury", "lead", "cadmium")]
    expect_identical(plsi_bootstrap(fit, frame, data$z, data$y, B = 3, n_jobs = 2)$bootstrap$draws,
                     boot$draws)
    expect_identical(plsi_bootstrap(fit, unname(data$x), unname(data$z), data$y, B = 3,
                                    n_jobs = 2)$bootstrap$draws, boot$draws)
    other <- data$x
    colnames(other)[3] <- "arsenic"
    expect_error(plsi_bootstrap(fit, other, data$z, data$y, B = 3),
                 "x has some of the fitted column names but not mercury")
  }
})

test_that("a custom grid and explicit data are used", {
  data <- simulate()
  fit <- splineplsi(data$x, data$z, data$y)
  boot <- plsi_bootstrap(fit, data$x, data$z, data$y, B = 4, n_jobs = 2, g_grid = c(-1, 0, 1))
  expect_identical(boot$bootstrap$g_grid, c(-1, 0, 1))
  expect_identical(dim(boot$bootstrap$g_draws), c(4L, 3L))
  expect_identical(boot$bootstrap$draws, plsi_bootstrap(fit, B = 4, n_jobs = 2)$bootstrap$draws)
})
