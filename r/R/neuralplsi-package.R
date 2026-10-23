#' neuralplsi: Neural network partial-linear single-index models
#'
#' Fits \eqn{g(x'\beta) + z'\gamma} for continuous, binary and right-censored
#' outcomes with a neural-network link (\code{\link{neuralplsi}}) or a B-spline
#' link (\code{\link{splineplsi}}), with pairs-bootstrap inference
#' (\code{\link{plsi_bootstrap}}). The C++ engine is the one used by the Python
#' package, so fits, bootstrap draws and model files
#' (\code{\link{save_model}}) agree exactly across the two languages on the
#' same machine.
#'
#' @references Do, H., Wang, Y., Liu, M. and Lee, M. (2025). Neural
#'   network-based partial-linear single-index models for environmental
#'   mixtures analysis. arXiv:2512.11593.
#' @keywords internal
#' @useDynLib neuralplsi, .registration = TRUE
#' @importFrom stats coef confint predict
#' @importFrom graphics plot
"_PACKAGE"

engine_version <- function() .Call(C_version)

.onLoad <- function(libname, pkgname) {
  package <- getNamespaceVersion(pkgname)[[1L]]
  if (!identical(engine_version(), package))
    stop(sprintf("The compiled engine is version %s, but the package is version %s",
                 engine_version(), package), call. = FALSE)
}
