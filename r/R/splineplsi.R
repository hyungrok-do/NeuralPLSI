#' Fit a SplinePLSI model
#'
#' Fits \eqn{g(x'\beta) + z'\gamma} with a B-spline link \eqn{g}, in double
#' precision, with the C++ engine shared with Python's \code{SplinePLSI}.
#'
#' For each unit \eqn{\beta}, \code{num_knots} knots sit at equally spaced order
#' statistics of \eqn{x'\beta} (Wang et al., 2020), clamped at the boundary
#' knots, and \eqn{\gamma} and the spline coefficients are the exact
#' ridge-penalized fit (\code{alpha}) on \eqn{[z, B(x'\beta)]}: least squares,
#' logistic regression or the Breslow partial likelihood. \eqn{\beta} is
#' profiled over the unit sphere by BFGS from a seeded random direction,
#' followed by a compass search, with \eqn{\beta_1 \ge 0}. \code{max_iter} bounds
#' the BFGS iterations, which stop when a step improves the profile loss by less
#' than \code{tol}. The basis sums to one, so \eqn{g} carries the intercept; for
#' Cox outcomes \eqn{g} is shifted to \eqn{g(0) = 0}. Outside the boundary
#' knots \eqn{g} is constant.
#'
#' @inheritParams neuralplsi
#' @param num_knots Number of knots (at least 2).
#' @param degree Spline degree.
#' @param alpha Ridge penalty on \eqn{\gamma} and the spline coefficients.
#' @param max_iter Maximum BFGS iterations.
#' @param tol Convergence tolerance on the profile loss.
#' @param seed Integer seed in \eqn{[0, 2^{53}]} for the starting direction.
#' @return An object of class \code{c("splineplsi", "plsi")}: a list with
#'   \code{coefficients}, \code{beta}, \code{gamma}, \code{spline} (the spline
#'   coefficients), \code{knots}, \code{family}, \code{settings},
#'   \code{diagnostics}, \code{data}, \code{seed} and \code{call}.
#' @references Wang, Y., Wu, Y., Jacobson, M. H., et al. (2020). A family of
#'   partial-linear single-index models for analyzing complex environmental
#'   exposures with continuous, categorical, time-to-event, and longitudinal
#'   health outcomes. \emph{Environmental Health}, 19, 96.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(600), 200, 3)
#' z <- matrix(rnorm(200), 200, 1)
#' g <- function(u) 5 * (plogis(2 * u) - 0.5)
#' y <- rbinom(200, 1, plogis(g(drop(x %*% c(2, 2, -1) / 3)) + z[, 1]))
#' fit <- splineplsi(x, z, y, family = "binary")
#' summary(fit)
#' @export
splineplsi <- function(x, z, y, family = c("continuous", "binary", "cox"), num_knots = 5,
                       degree = 3, alpha = 1e-6, max_iter = 200, tol = 1e-6, seed = 0) {
  call <- match.call()
  family <- match.arg(family)
  num_knots <- as_count(num_knots, "num_knots", 2)
  degree <- as_count(degree, "degree", 0)
  max_iter <- as_count(max_iter, "max_iter", 1)
  alpha <- as_number(alpha, "alpha")
  tol <- as_number(tol, "tol")
  seed <- as_seed(seed)
  data <- model_data(x, z, y, family)
  settings <- list(num_knots = num_knots, degree = degree, alpha = alpha, max_iter = max_iter,
                   tol = tol)
  result <- .Call(C_spline_fit, spline_config(family, settings, data), c(alpha, tol),
                  data$x, data$z, data$y, seed)
  beta <- stats::setNames(result$beta, colnames(data$x))
  gamma <- stats::setNames(result$gamma, colnames(data$z))
  diagnostics <- list(loss = result$diagnostics[1L], iterations = as.integer(result$diagnostics[2L]),
                      converged = result$diagnostics[3L] == 1)
  structure(list(coefficients = c(beta, gamma), beta = beta, gamma = gamma,
                 spline = result$spline, knots = result$knots, family = family,
                 settings = settings, diagnostics = diagnostics, data = data, seed = seed,
                 call = call),
            class = c("splineplsi", "plsi"))
}

spline_config <- function(family, settings, data) {
  as.integer(c(ncol(data$x), ncol(data$z), families[[family]], settings$num_knots,
               settings$degree, settings$max_iter))
}

spline_basis <- function(object, index) {
  t(.Call(C_spline_basis, object$knots, object$settings$degree, as.double(index)))
}
