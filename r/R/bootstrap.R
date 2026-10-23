#' Pairs-bootstrap inference
#'
#' Resamples rows with replacement and refits the model to each resample in
#' C++ on native threads, then attaches the draws, standard errors and
#' percentile intervals to the fit, where \code{summary()}, \code{confint()} and
#' \code{plot()} use them.
#'
#' Replicate \eqn{b = 0, \dots, B-1} resamples rows with the engine's
#' \code{splitmix64_bootstrap_v1} generator seeded by \code{seed + b} and refits
#' from a fresh initialization with fit seed \code{seed + 1337 + b}, using the
#' same procedure as the original fit. Draws therefore do not depend on
#' \code{n_jobs} or on \code{B}, NeuralPLSI and SplinePLSI see identical
#' resamples for the same seed, and the draws equal those of Python's
#' \code{inference_bootstrap} with \code{random_state = seed}. A failed refit
#' (for example a Cox resample without events) fails the whole call. The native
#' call cannot be interrupted from R.
#'
#' The name avoids \code{boot::boot()} and \code{bootstrap::bootstrap()}.
#'
#' @param object A \code{neuralplsi} or \code{splineplsi} fit.
#' @param x,z,y Data to resample, by default the training data kept in the fit.
#'   A model read with \code{\link{load_model}} keeps no data, so pass them.
#' @param B Number of bootstrap draws (at least 2).
#' @param seed Integer seed in \eqn{[0, 2^{53}]}.
#' @param n_jobs Native threads: -1 for all available CPUs, -2 for all but one,
#'   or a positive limit. Capped by CPU affinity, cgroup quotas,
#'   \code{SLURM_CPUS_PER_TASK} and \code{B}.
#' @param level Confidence level of the percentile intervals.
#' @param g_grid Index values at which to bootstrap \eqn{g}. \code{NULL} uses
#'   101 equally spaced values over the range of the fitted index
#'   \eqn{x'\hat\beta}. Coefficient draws do not depend on the grid.
#' @param quantile Percentile interpolation: \code{"linear"} (type 7) or
#'   \code{"weibull"} (type 6). It changes the intervals, not the draws.
#' @param ... Unused.
#' @return The fit, with a \code{bootstrap} element: a list with \code{B},
#'   \code{seed}, \code{level}, \code{quantile}, \code{workers} (threads used),
#'   \code{protocol}, \code{draws} (a B-row matrix of coefficient draws),
#'   \code{summary} (mean, sample SD \code{se}, \code{lower}, \code{upper} per
#'   coefficient), \code{g_grid}, \code{g_draws} and \code{g_summary}, and for
#'   SplinePLSI \code{spline_draws} and \code{spline_summary}.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(600), 200, 3)
#' z <- matrix(rnorm(200), 200, 1)
#' y <- sin(x %*% c(0.6, -0.6, 0.5)) + 0.5 * z[, 1] + rnorm(200, sd = 0.2)
#' fit <- plsi_bootstrap(neuralplsi(x, z, y), B = 20, n_jobs = 2)
#' summary(fit)
#' confint(fit, level = 0.9)
#' @export
plsi_bootstrap <- function(object, ...) UseMethod("plsi_bootstrap")

#' @rdname plsi_bootstrap
#' @export
plsi_bootstrap.plsi <- function(object, x = object$data$x, z = object$data$z,
                                y = object$data$y, B = 1000, seed = 0, n_jobs = -1,
                                level = 0.95, g_grid = NULL,
                                quantile = c("linear", "weibull"), ...) {
  if (is.null(x)) stored_data(object)
  quantile <- match.arg(quantile)
  B <- as_count(B, "B", 2)
  seed <- as_seed(seed)
  n_jobs <- as_jobs(n_jobs)
  level <- as_level(level)
  data <- model_data(x, z, y, object$family)
  if (ncol(data$x) != length(object$beta) || ncol(data$z) != length(object$gamma))
    stop("Bootstrap dimensions differ from the fitted model", call. = FALSE)
  if (is.null(g_grid)) {
    g_grid <- range(data$x %*% object$beta)
    g_grid <- seq(g_grid[1L], g_grid[2L], length.out = 101L)
  }
  if (!is.numeric(g_grid) || !length(g_grid) || !all(is.finite(g_grid)))
    stop("g_grid must be nonempty and finite", call. = FALSE)
  g_grid <- as.double(g_grid)
  method <- as.integer(quantile == "weibull")
  result <- if (inherits(object, "neuralplsi")) {
    .Call(C_bootstrap, engine(object), data$x, data$z, data$y, B, seed, n_jobs, g_grid,
          level, method)
  } else {
    .Call(C_spline_bootstrap, spline_config(object$family, object$settings, data),
          c(object$settings$alpha, object$settings$tol), data$x, data$z, data$y, B, seed,
          n_jobs, g_grid, level, method)
  }
  statistics <- c("mean", "se", "lower", "upper")
  draws <- t(result$draws)
  summary <- result$summary
  colnames(summary) <- statistics
  k <- length(object$coefficients)
  s <- length(object$spline)
  g <- k + s + seq_along(g_grid)
  names <- names(object$coefficients)
  boot <- list(B = B, seed = seed, level = level, quantile = quantile, workers = result$workers,
               protocol = bootstrap_protocol,
               draws = draws[, seq_len(k), drop = FALSE],
               summary = summary[seq_len(k), , drop = FALSE],
               g_grid = g_grid, g_draws = draws[, g, drop = FALSE],
               g_summary = summary[g, , drop = FALSE])
  dimnames(boot$draws) <- list(NULL, names)
  rownames(boot$summary) <- names
  if (s) {
    boot$spline_draws <- draws[, k + seq_len(s), drop = FALSE]
    boot$spline_summary <- summary[k + seq_len(s), , drop = FALSE]
  }
  object$bootstrap <- boot
  object
}

# Bootstrap summaries at another level, from the stored draws, with the engine's
# percentile rule.
draw_summary <- function(draws, level, quantile) {
  summary <- .Call(C_bootstrap_summary, t(draws), as_level(level),
                   as.integer(quantile == "weibull"))
  dimnames(summary) <- list(colnames(draws), c("mean", "se", "lower", "upper"))
  summary
}
