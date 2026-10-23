test_that("fits survive saveRDS and readRDS", {
  data <- simulate("binary")
  fit <- plsi_bootstrap(small_fit(data, "binary"), B = 4, n_jobs = 2)
  spline <- splineplsi(data$x, data$z, data$y, family = "binary")
  file <- tempfile(fileext = ".rds")
  on.exit(unlink(file))
  saveRDS(list(fit, spline), file)
  restored <- readRDS(file)
  copy <- restored[[1]]
  # identical() compares external pointers by address; a deserialized one is NULL.
  null <- unserialize(serialize(fit$pointer, NULL))
  expect_true(identical(copy$pointer, null))
  expect_identical(predict(copy, type = "response"), predict(fit, type = "response"))
  expect_false(identical(restored[[1]]$pointer, null))
  expect_identical(g_function(copy, c(-1, 0, 1)), g_function(fit, c(-1, 0, 1)))
  expect_identical(plsi_bootstrap(copy, B = 4, n_jobs = 2)$bootstrap, fit$bootstrap)
  expect_identical(confint(copy), confint(fit))
  expect_identical(predict(restored[[2]]), predict(spline))
})

test_that("save_model and load_model round trip the engine model", {
  for (family in all_families) {
    data <- simulate(family)
    fit <- small_fit(data, family, add_intercept = family != "cox")
    file <- tempfile(fileext = ".nplsi")
    save_model(fit, file)
    expect_identical(readLines(file, n = 1), "NEURALPLSI_NATIVE 5")
    loaded <- load_model(file)
    unlink(file)
    expect_identical(unname(coef(loaded)), unname(coef(fit)))
    expect_identical(names(loaded$beta), c("x1", "x2", "x3"))
    expect_identical(loaded$settings, fit$settings)
    expect_identical(loaded$diagnostics, fit$diagnostics)
    expect_identical(loaded$state, fit$state)
    expect_null(loaded$data)
    for (type in c("link", "response", "risk", "index"))
      expect_identical(predict(loaded, unname(data$x), unname(data$z), type = type),
                       predict(fit, type = type))
    expect_error(predict(loaded), "keeps no training data")
    expect_error(plsi_bootstrap(loaded), "keeps no training data")
    expect_identical(plsi_bootstrap(loaded, data$x, data$z, data$y, B = 3, n_jobs = 2)$bootstrap$draws,
                     unname(plsi_bootstrap(fit, B = 3, n_jobs = 2)$bootstrap$draws),
                     ignore_attr = TRUE)
    # A blank name is not matched to the loaded model's x2: columns go by position.
    blank <- data$x
    colnames(blank)[2] <- ""
    expect_identical(plsi_bootstrap(loaded, blank, data$z, data$y, B = 3, n_jobs = 2)$bootstrap$draws,
                     plsi_bootstrap(loaded, data$x, data$z, data$y, B = 3, n_jobs = 2)$bootstrap$draws)
  }
})

test_that("model file errors are reported", {
  data <- simulate()
  expect_error(save_model(splineplsi(data$x, data$z, data$y), tempfile()), "saveRDS")
  expect_error(load_model(tempfile()), "Could not open native model")
  file <- tempfile()
  writeLines("NEURALPLSI_NATIVE 4", file)
  expect_error(load_model(file), "Unsupported model state format")
  unlink(file)
  expect_error(save_model(small_fit(data, "continuous"), file.path(tempfile(), "missing", "m.nplsi")),
               "Could not write native model|No such file")
})

test_that("finalizers release engine models", {
  data <- simulate(n = 40)
  for (i in 1:20) small_fit(data, "continuous", max_iter = 5)
  expect_silent(gc())
})
