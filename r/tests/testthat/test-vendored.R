test_that("the compiled engine has the package version", {
  expect_identical(.Call(neuralplsi:::C_version), getNamespaceVersion("neuralplsi")[[1]])
})

test_that("the vendored engine sources are byte-identical to core/", {
  root <- checkout()
  skip_if(is.null(root), "not run from a NeuralPLSI checkout")
  bytes <- function(path) readBin(path, "raw", file.size(path))
  for (file in c("src/neuralplsi.cpp", "src/spline.cpp", "include/neuralplsi.h")) {
    expect_identical(bytes(file.path(root, "r", "src", basename(file))),
                     bytes(file.path(root, "core", file)),
                     label = paste0("r/src/", basename(file), " (run r/tools/sync-core.sh)"))
  }
  version <- read.dcf(file.path(root, "r", "DESCRIPTION"), "Version")[[1]]
  header <- grep("^#define NEURALPLSI_VERSION ", readLines(file.path(root, "core", "include",
                                                                     "neuralplsi.h")), value = TRUE)
  python <- grep("^__version__ = ", readLines(file.path(root, "python", "neuralplsi",
                                                        "_version.py")), value = TRUE)
  expect_identical(header, sprintf("#define NEURALPLSI_VERSION \"%s\"", version))
  expect_identical(python, sprintf("__version__ = '%s'", version))
})
