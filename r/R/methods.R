model_name <- function(object) if (inherits(object, "neuralplsi")) "NeuralPLSI" else "SplinePLSI"

blocks <- function(object) {
  c(rep("beta", length(object$beta)), rep("gamma", length(object$gamma)),
    if (!is.null(object$intercept)) "intercept")
}

percent <- function(level) {
  paste(format(100 * c(1 - level, 1 + level) / 2, trim = TRUE, scientific = FALSE, digits = 3), "%")
}

fit_line <- function(object) {
  d <- object$diagnostics
  if (inherits(object, "splineplsi"))
    return(sprintf("BFGS: %d iterations, %s, profile loss %s.", d$iterations,
                   if (d$converged) "converged" else "not converged", format(d$loss, digits = 6)))
  s <- object$settings
  network <- sprintf("%d hidden layer%s of %d %s units", s$n_hidden_layers,
                     if (s$n_hidden_layers > 1L) "s" else "", s$hidden_units, s$activation)
  if (s$optimizer == "lbfgs")
    sprintf("Network: %s. L-BFGS: %d iterations, penalized objective %s.", network,
            as.integer(d$iterations), format(d$objective, digits = 6))
  else
    sprintf("Network: %s. AdamW: %d epochs, best validation loss %s at epoch %d.", network,
            as.integer(d$iterations), format(d$objective, digits = 6), as.integer(d$best_epoch))
}

#' Methods for NeuralPLSI and SplinePLSI fits
#'
#' \code{coef()} returns \eqn{\beta}, \eqn{\gamma} and, if fitted, the
#' intercept, labelled by the column names of \code{x} and \code{z}.
#' \code{summary()} tabulates them with bootstrap standard errors and percentile
#' intervals once \code{\link{plsi_bootstrap}} has run. \code{confint()} returns
#' those intervals; another \code{level} is computed from the stored draws.
#'
#' @param object,x A \code{neuralplsi} or \code{splineplsi} fit, or its summary.
#' @param parm Coefficient names or indices; all by default.
#' @param level Confidence level; by default that of the bootstrap.
#' @param digits Significant digits to print.
#' @param ... Unused.
#' @return \code{coef()}: a named numeric vector. \code{summary()}: an object of
#'   class \code{summary.plsi} whose \code{coefficients} matrix has columns
#'   Estimate, Std. Error and the interval bounds. \code{confint()}: a matrix
#'   with one row per coefficient.
#' @name plsi-methods
NULL

#' @rdname plsi-methods
#' @export
coef.plsi <- function(object, ...) object$coefficients

#' @rdname plsi-methods
#' @export
print.plsi <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  n <- if (is.null(x$data)) NA else nrow(x$data$x)
  count <- function(k, noun) sprintf("%d %s%s", k, noun, if (k == 1L) "" else "s")
  cat(sprintf("%s, %s outcome%s, %s, %s\n", model_name(x), x$family,
              if (is.na(n)) "" else sprintf(", n = %d", n), count(length(x$beta), "exposure"),
              count(length(x$gamma), "covariate")))
  if (!is.null(x$call)) cat("Call: ", paste(deparse(x$call), collapse = "\n"), "\n", sep = "")
  cat("\nIndex coefficients (beta):\n")
  print(x$beta, digits = digits)
  if (length(x$gamma)) {
    cat("\nCovariate coefficients (gamma):\n")
    print(x$gamma, digits = digits)
  }
  if (!is.null(x$intercept)) cat("\nIntercept:", format(x$intercept, digits = digits), "\n")
  cat("\n", fit_line(x), "\n", sep = "")
  if (!is.null(x$bootstrap))
    cat(sprintf("Bootstrap: B = %d; see summary() and confint().\n", x$bootstrap$B))
  invisible(x)
}

#' @rdname plsi-methods
#' @export
summary.plsi <- function(object, ...) {
  estimate <- coef(object)
  boot <- object$bootstrap
  table <- cbind(Estimate = estimate)
  if (!is.null(boot)) {
    table <- cbind(table, boot$summary[names(estimate), c("se", "lower", "upper"), drop = FALSE])
    colnames(table) <- c("Estimate", "Std. Error", percent(boot$level))
  }
  structure(list(model = model_name(object), family = object$family, call = object$call,
                 n = if (is.null(object$data)) NA_integer_ else nrow(object$data$x),
                 coefficients = table, blocks = blocks(object), fit = fit_line(object),
                 bootstrap = boot[c("B", "seed", "level", "quantile", "workers", "protocol")]),
            class = "summary.plsi")
}

#' @rdname plsi-methods
#' @export
print.summary.plsi <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("%s, %s outcome%s\n", x$model, x$family,
              if (is.na(x$n)) "" else sprintf(", n = %d", x$n)))
  if (!is.null(x$call)) cat("Call: ", paste(deparse(x$call), collapse = "\n"), "\n", sep = "")
  titles <- c(beta = "Index coefficients (beta, unit length):",
              gamma = "Covariate coefficients (gamma):", intercept = "Intercept:")
  for (block in unique(x$blocks)) {
    cat("\n", titles[[block]], "\n", sep = "")
    print(x$coefficients[x$blocks == block, , drop = FALSE], digits = digits)
  }
  cat("\n", x$fit, "\n", sep = "")
  b <- x$bootstrap
  if (length(b))
    cat(sprintf(paste("Bootstrap: B = %d pairs resamples with cold-start refits, seed %s,",
                      "%s%% percentile intervals (%s quantiles), %d thread%s.\n"),
                b$B, format(b$seed, scientific = FALSE), format(100 * b$level), b$quantile,
                b$workers, if (b$workers == 1L) "" else "s"))
  else
    cat("No bootstrap: run plsi_bootstrap() for standard errors and intervals.\n")
  invisible(x)
}

#' @rdname plsi-methods
#' @export
confint.plsi <- function(object, parm, level = NULL, ...) {
  boot <- object$bootstrap
  if (is.null(boot)) stop("No bootstrap results: run plsi_bootstrap() first", call. = FALSE)
  if (is.null(level)) level <- boot$level
  summary <- if (identical(as_level(level), boot$level)) boot$summary
             else draw_summary(boot$draws, level, boot$quantile)
  intervals <- summary[, c("lower", "upper"), drop = FALSE]
  colnames(intervals) <- percent(level)
  if (missing(parm)) intervals else intervals[parm, , drop = FALSE]
}

#' Predictions from a NeuralPLSI or SplinePLSI fit
#'
#' NeuralPLSI predictions come from the engine in float32 and equal Python's.
#' SplinePLSI predictions are computed in double precision from the engine's
#' spline basis with R's matrix products, which sum in a different order from
#' numpy's: they agree with Python's to rounding error (absolute differences of
#' a few times \eqn{10^{-15}} on the package's test data), not bit for bit.
#'
#' @param object A \code{neuralplsi} or \code{splineplsi} fit.
#' @param newx,newz New exposures and covariates, with as many columns as the
#'   fit. Columns are matched to the fit by name when \code{newx} or \code{newz}
#'   has any of the fitted names, and it must then have all of them; by position
#'   when it has none of them or no column names. Without \code{newx}, the
#'   training data are used, and \code{newz} must be omitted too. \code{newz} is
#'   not needed for \code{type = "index"} or without covariates.
#' @param type \code{"link"}: the linear predictor \eqn{g(x'\beta) + z'\gamma}
#'   (plus the intercept). \code{"response"}: the linear predictor for continuous
#'   outcomes, the probability for binary outcomes, and the relative risk
#'   \eqn{\exp} of the linear predictor for Cox outcomes. \code{"risk"}: the
#'   exponentiated linear predictor (the partial hazard). \code{"index"}:
#'   \eqn{g(x'\beta)} alone. These are the engine's prediction modes 0 to 3.
#' @param ... Unused.
#' @return A numeric vector with one value per row.
#' @export
predict.plsi <- function(object, newx, newz = NULL, type = c("link", "response", "risk", "index"),
                         ...) {
  type <- match.arg(type)
  if (missing(newx)) {
    if (!is.null(newz)) stop("newz given without newx", call. = FALSE)
    data <- stored_data(object)
    newx <- data$x
    newz <- data$z
  }
  x <- new_data(newx, names(object$beta), "newx")
  z <- matrix(0, nrow(x), 0L)
  if (type != "index" && length(object$gamma)) {
    if (is.null(newz)) stop("newz is required for this model", call. = FALSE)
    z <- new_data(newz, names(object$gamma), "newz")
    if (nrow(z) != nrow(x)) stop("newx and newz must have the same number of rows", call. = FALSE)
  }
  mode <- switch(type, link = 0L, risk = 2L, index = 3L,
                 response = c(continuous = 0L, binary = 1L, cox = 2L)[[object$family]])
  if (inherits(object, "neuralplsi")) return(.Call(C_predict, engine(object), x, z, mode))
  g <- drop(spline_basis(object, x %*% object$beta) %*% object$spline)
  if (mode == 3L) return(g)
  eta <- g + drop(z %*% object$gamma)
  switch(mode + 1L, eta, stats::plogis(eta), exp(eta))
}

#' Evaluate the link function
#'
#' @param object A \code{neuralplsi} or \code{splineplsi} fit.
#' @param index Numeric vector of index values \eqn{x'\beta}.
#' @return \eqn{g} at \code{index}: float32 values from the engine for
#'   NeuralPLSI, double precision for SplinePLSI.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(600), 200, 3)
#' g <- function(u) 5 * (plogis(2 * u) - 0.5)
#' y <- g(drop(x %*% c(2, 2, -1) / 3)) + rnorm(200, sd = 0.5)
#' fit <- splineplsi(x, NULL, y)
#' g_function(fit, seq(-1, 1, by = 0.5))
#' @export
g_function <- function(object, index) {
  if (!inherits(object, "plsi")) stop("object must be a neuralplsi or splineplsi fit", call. = FALSE)
  if (!is.numeric(index) || !all(is.finite(index)))
    stop("index must be numeric and finite", call. = FALSE)
  index <- as.double(index)
  if (inherits(object, "neuralplsi")) return(.Call(C_link, engine(object), index))
  drop(spline_basis(object, index) %*% object$spline)
}

#' Plot the estimated link function
#'
#' Draws \eqn{g} over a grid of index values, with the pointwise bootstrap
#' percentile band when \code{\link{plsi_bootstrap}} has run, and a rug of the
#' fitted index values of the training data.
#'
#' @param x A \code{neuralplsi} or \code{splineplsi} fit.
#' @param grid Index values. By default the bootstrap grid, or 201 values over
#'   the range of the fitted index. The band is drawn on the bootstrap grid only.
#' @param band Draw the bootstrap band if available.
#' @param rug Mark the fitted index values of the training data.
#' @param xlab,ylab,main Axis labels and title.
#' @param band_col Fill colour of the band.
#' @param ... Passed to \code{plot()}.
#' @return Invisibly, a data frame with the grid, \eqn{g} and, with a band, its
#'   lower and upper bounds.
#' @export
plot.plsi <- function(x, grid = NULL, band = TRUE, rug = TRUE, xlab = "Index x'beta",
                      ylab = "g(index)", main = model_name(x), band_col = "grey85", ...) {
  boot <- x$bootstrap
  index <- if (is.null(x$data)) NULL else drop(x$data$x %*% x$beta)
  if (is.null(grid)) {
    grid <- if (!is.null(boot)) boot$g_grid
            else if (!is.null(index)) seq(min(index), max(index), length.out = 201L)
            else stop("This model keeps no training data: pass grid", call. = FALSE)
  }
  g <- g_function(x, grid)
  result <- data.frame(index = grid, g = g)
  shade <- band && !is.null(boot) && identical(as.double(grid), boot$g_grid)
  if (shade) {
    result$lower <- boot$g_summary[, "lower"]
    result$upper <- boot$g_summary[, "upper"]
  }
  limits <- range(result[-1L])
  graphics::plot(grid, g, type = "n", ylim = limits, xlab = xlab, ylab = ylab, main = main, ...)
  if (shade)
    graphics::polygon(c(grid, rev(grid)), c(result$lower, rev(result$upper)), col = band_col,
                      border = NA)
  graphics::lines(grid, g, lwd = 2)
  if (rug && !is.null(index)) graphics::rug(index[index >= min(grid) & index <= max(grid)])
  invisible(result)
}
