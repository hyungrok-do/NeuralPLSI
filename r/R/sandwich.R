#' Sandwich standard errors
#'
#' Standard errors and Wald intervals for \eqn{\beta}, \eqn{\gamma} and the
#' intercept from the curvature of the fitted objective, with no refitting:
#' the variance of its minimizer, \eqn{H^{-1} C H^{-1} / n^2}, with \eqn{H} the
#' Hessian of the objective and \eqn{C} the sum of the rows' centered score
#' outer products, computed in C++ in double precision. \eqn{\beta} enters in
#' the tangent plane of the unit sphere. For Cox outcomes the rows' scores are
#' Lin and Wei's score residuals of the Breslow partial likelihood.
#'
#' \code{method = "full"} takes \eqn{H} and \eqn{C} over every weight of the
#' network, so the network's estimation enters the variance of \eqn{\beta} and
#' \eqn{\gamma}; the Hessian is dense, for networks of at most 8000 weights, and
#' its columns use \code{n_jobs} threads. \code{method = "frozen"} takes them
#' over \eqn{\beta}, \eqn{\gamma} and the intercept with the network held
#' fixed: any network size, slightly narrower (in simulation 1--7\% at
#' converged L-BFGS fits). \code{"auto"} is \code{"frozen"} for
#' \code{optimizer = "adam_frozen"} and \code{"full"} otherwise.
#'
#' The formula describes a minimizer. The result reports how far the fit is from
#' one: the largest component of the remaining Newton step \eqn{H^{-1}\nabla Q}
#' for \eqn{\beta} and for \eqn{\gamma}, in SE units (well below 1 at a converged
#' fit), and the Hessian's eigenvalue nearest zero.
#'
#' @inheritParams plsi_bootstrap
#' @param object A \code{neuralplsi} fit.
#' @param method \code{"auto"}, \code{"full"} or \code{"frozen"}.
#' @param level Confidence level of the Wald intervals.
#' @param n_jobs Native threads for the full method's Hessian, as in
#'   \code{\link{plsi_bootstrap}}.
#' @return The fit, with a \code{sandwich} element: a list with \code{method},
#'   \code{level}, \code{summary} (estimate, \code{se}, \code{lower},
#'   \code{upper} per coefficient), \code{newton_beta}, \code{newton_gamma},
#'   \code{min_eigenvalue} and \code{gradient_norm}.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(600), 200, 3)
#' z <- matrix(rnorm(200), 200, 1)
#' g <- function(u) 5 * (plogis(2 * u) - 0.5)
#' y <- g(drop(x %*% c(2, 2, -1) / 3)) + 0.5 * z[, 1] + rnorm(200, sd = 0.5)
#' fit <- plsi_sandwich(neuralplsi(x, z, y, hidden_units = 8), n_jobs = 2)
#' summary(fit)
#' @export
plsi_sandwich <- function(object, ...) UseMethod("plsi_sandwich")

#' @rdname plsi_sandwich
#' @export
plsi_sandwich.neuralplsi <- function(object, x = object$data$x, z = object$data$z,
                                     y = object$data$y, method = c("auto", "full", "frozen"),
                                     level = 0.95, n_jobs = -1, ...) {
  if (is.null(x)) stored_data(object)
  method <- match.arg(method)
  if (method == "auto") method <- if (object$settings$optimizer == "adam_frozen") "frozen" else "full"
  n_jobs <- as_jobs(n_jobs)
  level <- as_level(level)
  data <- model_data(x, z, y, object$family)
  if (ncol(data$x) != length(object$beta) || ncol(data$z) != length(object$gamma))
    stop("Sandwich dimensions differ from the fitted model", call. = FALSE)
  data$x <- fitted_columns(data$x, names(object$beta), "x", colnames(x))
  data$z <- fitted_columns(data$z, names(object$gamma), "z", colnames(z))
  result <- .Call(C_sandwich, engine(object), data$x, data$z, data$y,
                  as.integer(method == "frozen"), n_jobs)
  estimate <- coef(object)
  se <- stats::setNames(result$se, names(estimate))
  object$sandwich <- list(method = method, level = level, summary = wald_summary(estimate, se, level),
                          newton_beta = result$diagnostics[1L], newton_gamma = result$diagnostics[2L],
                          min_eigenvalue = result$diagnostics[3L], gradient_norm = result$diagnostics[4L])
  object
}

#' @rdname plsi_sandwich
#' @export
plsi_sandwich.splineplsi <- function(object, ...) {
  stop("plsi_sandwich() is available for NeuralPLSI fits; use plsi_bootstrap() for SplinePLSI",
       call. = FALSE)
}

# Estimate, SE and Wald interval per coefficient.
wald_summary <- function(estimate, se, level) {
  z <- stats::qnorm((1 + level) / 2)
  summary <- cbind(mean = estimate, se = se, lower = estimate - z * se, upper = estimate + z * se)
  rownames(summary) <- names(estimate)
  summary
}
