test_that("plsi_sandwich gives SEs, Wald intervals and the Python package's numbers", {
  data <- simulate("binary")
  # The package's iteration cap (the helper's 300 stop L-BFGS short of its rule).
  fit <- small_fit(data, "binary", add_intercept = TRUE, max_iter = 5000)
  expect_error(confint(fit, inference = "sandwich"), "run plsi_sandwich")
  fit <- plsi_sandwich(fit, level = 0.9, n_jobs = 1)
  s <- fit$sandwich
  expect_identical(s$method, "full")
  expect_identical(rownames(s$summary), names(coef(fit)))
  expect_true(all(is.finite(s$summary[, "se"]) & s$summary[, "se"] > 0))
  expect_equal(unname(s$summary[, "lower"]),
               unname(coef(fit) - stats::qnorm(0.95) * s$summary[, "se"]))
  expect_lt(s$newton_beta, 0.1)
  # Threads split the Hessian's columns without changing it.
  expect_identical(plsi_sandwich(fit, level = 0.9, n_jobs = 3)$sandwich$summary, s$summary)
  table <- summary(fit)$coefficients
  expect_identical(colnames(table), c("Estimate", "Std. Error", "5 %", "95 %"))
  expect_output(print(summary(fit)), "Sandwich \\(full\\): 90% Wald intervals")
  expect_identical(confint(fit), s$summary[, c("lower", "upper")] |> `colnames<-`(c("5 %", "95 %")))
  wider <- confint(fit, level = 0.99)
  expect_true(all(wider[, 1] < s$summary[, "lower"]))
  # With both, "auto" prefers the bootstrap; either can be asked for.
  both <- plsi_bootstrap(fit, B = 5, n_jobs = 1)
  expect_identical(colnames(summary(both)$coefficients)[3], "2.5 %")
  expect_identical(summary(both, inference = "sandwich")$coefficients[, "Std. Error"], s$summary[, "se"])
  frozen <- small_fit(data, "binary", optimizer = "adam_frozen", max_iter = 20, batch_size = 32)
  expect_identical(plsi_sandwich(frozen)$sandwich$method, "frozen")
})

test_that("plsi_sandwich handles Cox outcomes and refuses SplinePLSI fits", {
  data <- simulate("cox")
  fit <- plsi_sandwich(small_fit(data, "cox"), n_jobs = 1)
  expect_true(all(is.finite(fit$sandwich$summary[, "se"]) & fit$sandwich$summary[, "se"] > 0))
  expect_error(plsi_sandwich(splineplsi(data$x, data$z, data$y, family = "cox")), "NeuralPLSI fits")
})
