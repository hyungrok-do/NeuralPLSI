#' Fit a NeuralPLSI model
#'
#' Fits the partial-linear single-index model \eqn{g(x'\beta) + z'\gamma}, where
#' the link \eqn{g} is a neural network, with the C++ engine shared with the
#' Python package. The defaults are those of Python's \code{NeuralPLSI}.
#'
#' Each fit uses all rows and minimizes the outcome loss (mean squared error,
#' binary cross-entropy, or the negative Breslow partial likelihood divided by
#' the number of events) plus \eqn{g(0)^2} and \code{weight_decay}/2 times the
#' squared link-network weights, by full-batch L-BFGS. \eqn{\beta} has unit
#' length and \eqn{\beta_1 \ge 0}. With \eqn{\beta} and \eqn{g} fixed,
#' \eqn{\gamma} and the optional intercept are then fitted exactly.
#'
#' \code{optimizer = "adam"} is a baseline for typical neural-network training,
#' not a recommended estimator: AdamW on minibatches of \code{batch_size} rows
#' of a random 80\% of the data, for at most \code{max_iter} epochs, stopping
#' after \code{patience} epochs without improvement on the other 20\%.
#'
#' Data are converted to float32 (round to nearest, as numpy's
#' \code{astype(float32)}), so a fit equals the Python fit of the same data
#' with the same seed. The fit keeps the training data for
#' \code{\link{plsi_bootstrap}}, \code{predict} and \code{plot}, and the
#' engine state, so it survives \code{saveRDS}/\code{readRDS}.
#'
#' @param x Numeric matrix or data frame of exposures (at least one column).
#' @param z Numeric matrix or data frame of covariates, or \code{NULL} for none.
#' @param y Outcome: a numeric vector (continuous), a 0/1 or logical vector
#'   (binary), or for \code{family = "cox"} a right-censored
#'   \code{survival::Surv} object or a two-column matrix of times and event
#'   indicators.
#' @param family \code{"continuous"}, \code{"binary"} or \code{"cox"}.
#' @param hidden_units Width of each hidden layer of the link network.
#' @param n_hidden_layers Number of hidden layers.
#' @param activation \code{"LeakyReLU"}, \code{"ELU"}, \code{"ReLU"} or \code{"Tanh"}.
#' @param weight_decay L2 penalty on the link network (decoupled weight decay for AdamW).
#' @param max_iter L-BFGS iterations, or AdamW epochs. 0 only initializes.
#' @param add_intercept Fit an intercept (not for Cox outcomes).
#' @param initial Start from a linear, logistic or Cox regression fit.
#' @param optimizer \code{"lbfgs"} or the \code{"adam"} baseline.
#' @param batch_size,learning_rate,patience AdamW settings.
#' @param seed Integer seed in \eqn{[0, 2^{53}]} for the initialization.
#' @return An object of class \code{c("neuralplsi", "plsi")}: a list with
#'   \code{coefficients}, \code{beta}, \code{gamma}, \code{intercept},
#'   \code{family}, \code{settings}, \code{diagnostics}, \code{data},
#'   \code{seed}, \code{call}, and the engine \code{state}.
#' @seealso \code{\link{plsi_bootstrap}}, \code{\link{predict.plsi}},
#'   \code{\link{save_model}}, \code{\link{splineplsi}}
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(600), 200, 3, dimnames = list(NULL, c("a", "b", "c")))
#' z <- cbind(age = rnorm(200))
#' y <- sin(x %*% c(0.6, -0.6, 0.5)) + 0.5 * z[, 1] + rnorm(200, sd = 0.2)
#' fit <- neuralplsi(x, z, y)
#' fit
#' head(predict(fit))
#' @export
neuralplsi <- function(x, z, y, family = c("continuous", "binary", "cox"),
                       hidden_units = 32, n_hidden_layers = 2,
                       activation = c("LeakyReLU", "ELU", "ReLU", "Tanh"),
                       weight_decay = 1e-3, max_iter = 5000, add_intercept = FALSE,
                       initial = FALSE, optimizer = c("lbfgs", "adam"), batch_size = 64,
                       learning_rate = 1e-3, patience = 10, seed = 0) {
  call <- match.call()
  family <- match.arg(family)
  activation <- match.arg(activation)
  optimizer <- match.arg(optimizer)
  add_intercept <- as_flag(add_intercept, "add_intercept")
  if (family == "cox" && add_intercept)
    stop("A Cox partial likelihood cannot identify an intercept", call. = FALSE)
  max_iter <- as_count(max_iter, "max_iter", 0)
  hidden_units <- as_count(hidden_units, "hidden_units", 1)
  n_hidden_layers <- as_count(n_hidden_layers, "n_hidden_layers", 1)
  batch_size <- as_count(batch_size, "batch_size", 1)
  patience <- as_count(patience, "patience", 1)
  weight_decay <- as_number(weight_decay, "weight_decay")
  learning_rate <- as_number(learning_rate, "learning_rate", positive = TRUE)
  initial <- as_flag(initial, "initial")
  seed <- as_seed(seed)
  data <- model_data(x, z, y, family)
  p <- ncol(data$x)
  q <- ncol(data$z)
  w <- hidden_units
  size <- p + q + 1 + 2 * w + (n_hidden_layers - 1) * (w * w + w) + w + 1 + add_intercept
  if (size > .Machine$integer.max) stop("The network is too large", call. = FALSE)
  config <- c(p, q, w, n_hidden_layers, add_intercept, families[[family]],
              activations[[activation]], size, max_iter, initial, optimizers[[optimizer]],
              batch_size, patience)
  pointer <- .Call(C_fit, as.integer(config), c(weight_decay, 1, learning_rate),
                   data$x, data$z, data$y, seed)
  fit <- neuralplsi_object(pointer, colnames(data$x), colnames(data$z))
  fit$data <- data
  fit$seed <- seed
  fit$call <- call
  fit
}

# The R object of an engine handle. Settings are read back from the engine, so
# fitted and loaded models are described the same way.
neuralplsi_object <- function(pointer, xnames = NULL, znames = NULL) {
  info <- .Call(C_info, pointer)
  config <- info$config
  p <- config[1L]
  q <- config[2L]
  if (is.null(xnames)) xnames <- sprintf("x%d", seq_len(p))
  if (is.null(znames)) znames <- sprintf("z%d", seq_len(q))
  beta <- stats::setNames(info$beta, xnames)
  gamma <- stats::setNames(info$gamma, znames)
  intercept <- if (config[5L]) info$intercept else NULL
  coefficients <- c(beta, gamma)
  if (!is.null(intercept)) coefficients <- c(coefficients, `(Intercept)` = intercept)
  settings <- list(hidden_units = config[3L], n_hidden_layers = config[4L],
                   activation = names(activations)[config[7L] + 1L],
                   weight_decay = info$hyper[1L], max_iter = config[9L],
                   add_intercept = config[5L] == 1L, initial = config[10L] == 1L,
                   optimizer = names(optimizers)[config[11L] + 1L], batch_size = config[12L],
                   learning_rate = info$hyper[3L], patience = config[13L])
  diagnostics <- stats::setNames(as.list(info$diagnostics),
                                 c("iterations", "gamma_iterations", "objective", "best_epoch"))
  structure(list(coefficients = coefficients, beta = beta, gamma = gamma, intercept = intercept,
                 family = names(families)[config[6L] + 1L], settings = settings,
                 diagnostics = diagnostics, data = NULL, seed = NULL, call = NULL,
                 pointer = pointer, state = info$state),
            class = c("neuralplsi", "plsi"))
}

# The live engine handle, rebuilt from the saved state after readRDS().
engine <- function(object) .Call(C_revive, object$pointer, object$state)
