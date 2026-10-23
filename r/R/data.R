# Argument and data checks mirror python/neuralplsi (base.validate_data,
# native.bootstrap_options and the model constructors).

families <- c(continuous = 0L, binary = 1L, cox = 2L)
activations <- c(ELU = 0L, LeakyReLU = 1L, ReLU = 2L, Tanh = 3L)
optimizers <- c(lbfgs = 0L, adam = 1L)
bootstrap_protocol <- "splitmix64_bootstrap_v1"

as_count <- function(value, name, minimum) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value != round(value) ||
      value < minimum || value > .Machine$integer.max)
    stop(sprintf("%s must be an integer >= %d", name, minimum), call. = FALSE)
  as.integer(value)
}

as_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value))
    stop(name, " must be TRUE or FALSE", call. = FALSE)
  value
}

as_number <- function(value, name, positive = FALSE) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value < 0 ||
      (positive && value == 0))
    stop(name, if (positive) " must be finite and positive" else " must be finite and nonnegative",
         call. = FALSE)
  as.double(value)
}

# Engine seeds are uint64; R doubles hold integers exactly up to 2^53.
as_seed <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value != round(value) ||
      value < 0 || value > 2^53)
    stop("seed must be an integer in [0, 2^53]", call. = FALSE)
  as.double(value)
}

as_jobs <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value != round(value) ||
      value == 0 || abs(value) > .Machine$integer.max)
    stop("n_jobs must be a nonzero 32-bit integer", call. = FALSE)
  as.integer(value)
}

as_level <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value <= 0 || value >= 1)
    stop("level must lie strictly between zero and one", call. = FALSE)
  as.double(value)
}

numeric_matrix <- function(value, name) {
  if (is.data.frame(value)) {
    if (!all(vapply(value, function(v) is.numeric(v) || is.logical(v), NA)))
      stop(name, " must have only numeric columns; code factors with model.matrix()", call. = FALSE)
    value <- as.matrix(value)
  }
  if (is.null(dim(value))) value <- matrix(value, ncol = 1L)
  if (!is.matrix(value) || !(is.numeric(value) || is.logical(value)))
    stop(name, " must be a numeric matrix or data frame", call. = FALSE)
  storage.mode(value) <- "double"
  value
}

column_names <- function(value, prefix) {
  names <- colnames(value)
  if (is.null(names)) names <- character(ncol(value))
  missing <- is.na(names) | names == ""
  names[missing] <- sprintf("%s%d", prefix, which(missing))
  names
}

outcome <- function(y, family) {
  if (family == "cox") {
    if (inherits(y, "Surv")) {
      if (!identical(attr(y, "type"), "right"))
        stop("Cox outcomes must be right-censored Surv objects", call. = FALSE)
      y <- unclass(y)[, 1:2, drop = FALSE]
    }
    y <- numeric_matrix(y, "y")
    if (ncol(y) != 2L)
      stop("Cox y must be a Surv object or a two-column matrix of times and event indicators",
           call. = FALSE)
    colnames(y) <- c("time", "status")
    return(y)
  }
  if (inherits(y, "Surv")) stop("Surv outcomes need family = \"cox\"", call. = FALSE)
  if (is.data.frame(y) || is.matrix(y)) {
    y <- numeric_matrix(y, "y")
    if (ncol(y) != 1L) stop("y must be a vector or a single column", call. = FALSE)
    y <- y[, 1L]
  }
  if (!is.numeric(y) && !is.logical(y)) stop("y must be numeric", call. = FALSE)
  as.double(y)
}

# x, z and y as double matrices (y a vector unless Cox), with column names.
model_data <- function(x, z, y, family) {
  if (is.null(x)) stop("x is required", call. = FALSE)
  x <- numeric_matrix(x, "x")
  z <- if (is.null(z)) matrix(0, nrow(x), 0L) else numeric_matrix(z, "z")
  y <- outcome(y, family)
  if (ncol(x) < 1L) stop("x must have at least one exposure column", call. = FALSE)
  n <- nrow(x)
  if (n < 2L || nrow(z) != n || NROW(y) != n)
    stop("x, z and y must have the same number of rows (at least two)", call. = FALSE)
  if (!all(is.finite(x)) || !all(is.finite(z)) || !all(is.finite(y)))
    stop("x, z and y must contain only finite values", call. = FALSE)
  if (family == "cox") {
    if (any(y[, 1L] < 0))
      stop("Cox y must contain nonnegative durations and event indicators", call. = FALSE)
    if (!all(y[, 2L] %in% c(0, 1)) || sum(y[, 2L]) == 0)
      stop("Cox outcomes need binary event indicators and at least one event", call. = FALSE)
  } else if (family == "binary" && !all(y %in% c(0, 1))) {
    stop("Binary outcomes must be coded 0 or 1", call. = FALSE)
  }
  xnames <- column_names(x, "x")
  znames <- column_names(z, "z")
  if (anyDuplicated(c(xnames, znames)))
    stop("The columns of x and z need distinct names", call. = FALSE)
  dimnames(x) <- list(NULL, xnames)
  dimnames(z) <- list(NULL, znames)
  list(x = x, z = z, y = y)
}

# New data for prediction, matched to the fitted columns by name when it has
# all of them, by position otherwise.
new_data <- function(value, names, label) {
  value <- if (is.null(value)) matrix(0, 0L, 0L) else numeric_matrix(value, label)
  if (!is.null(colnames(value)) && all(names %in% colnames(value)))
    value <- value[, names, drop = FALSE]
  if (ncol(value) != length(names))
    stop(sprintf("%s must have %d columns (%s)", label, length(names),
                 paste(names, collapse = ", ")), call. = FALSE)
  if (!all(is.finite(value))) stop(label, " must contain only finite values", call. = FALSE)
  value
}

stored_data <- function(object) {
  if (is.null(object$data))
    stop("This model keeps no training data (it was loaded from a file): pass x, z and y",
         call. = FALSE)
  object$data
}
