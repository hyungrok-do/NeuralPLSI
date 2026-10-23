# Runs from a checkout, with PYTHON or python3 able to import the checkout's
# python/neuralplsi; skipped otherwise, or fails if NEURALPLSI_REQUIRE_PYTHON=true.
test_that("R and Python give identical fits, bootstrap draws and model files", {
  unavailable <- function(message) {
    if (identical(Sys.getenv("NEURALPLSI_REQUIRE_PYTHON"), "true")) stop(message)
    skip(message)
  }
  root <- checkout()
  if (is.null(root)) unavailable("not run from a NeuralPLSI checkout")
  python <- Sys.getenv("PYTHON", Sys.which("python3"))
  if (!nzchar(python)) unavailable("no Python found")
  env <- paste0("PYTHONPATH=", shQuote(file.path(root, "python")))
  run <- function(...) suppressWarnings(system2(python, c(...), env = env, stdout = FALSE,
                                                stderr = FALSE))
  if (run("-c", shQuote("import neuralplsi")) != 0)
    unavailable(paste(python, "cannot import the checkout's neuralplsi package"))
  folder <- tempfile("crosscheck")
  dir.create(folder)
  on.exit(unlink(folder, recursive = TRUE))
  path <- function(family, name) file.path(folder, paste0(family, ".", name))
  write <- function(values, family, name) writeBin(as.double(values), path(family, name),
                                                   endian = "little")
  read <- function(family, name, columns = 1) {
    values <- readBin(path(family, name), "double", file.size(path(family, name)) / 8,
                      endian = "little")
    if (columns == 1) values else matrix(values, ncol = columns, byrow = TRUE)
  }
  data <- lapply(stats::setNames(all_families, all_families), simulate)
  for (family in all_families) {
    d <- data[[family]]
    writeLines(paste(nrow(d$x), ncol(d$x), ncol(d$z)), path(family, "dims"))
    write(t(d$x), family, "x")
    write(t(d$z), family, "z")
    write(if (family == "cox") t(d$y) else d$y, family, "y")
  }
  script <- test_path("crosscheck.py")
  expect_identical(run(shQuote(script), shQuote(folder), "fit"), 0L)
  d <- data$continuous
  expect_identical(unname(coef(neuralplsi(d$x, d$z, d$y, seed = 7))), read("default", "coef"))
  for (family in all_families) {
    d <- data[[family]]
    fit <- small_fit(d, family, seed = 7)
    expect_identical(unname(coef(fit)), read(family, "coef"))
    expect_identical(predict(fit), read(family, "link"))
    expect_identical(predict(load_model(path(family, "py.nplsi")), d$x, d$z), read(family, "link"))
    boot <- plsi_bootstrap(fit, B = 4, seed = 3, n_jobs = 2)$bootstrap
    expect_identical(unname(boot$draws), read(family, "draws", 5))
    save_model(fit, path(family, "r.nplsi"))
    spline <- splineplsi(d$x, d$z, d$y, family = family, seed = 7)
    expect_identical(unname(c(coef(spline), spline$spline)), read(family, "spline"))
    boot <- plsi_bootstrap(spline, B = 4, seed = 3, n_jobs = 2)$bootstrap
    expect_identical(unname(boot$draws), read(family, "spline_draws", 5))
  }
  expect_identical(run(shQuote(script), shQuote(folder), "load"), 0L)
  for (family in all_families)
    expect_identical(read(family, "r_link"), predict(small_fit(data[[family]], family, seed = 7)))
})
