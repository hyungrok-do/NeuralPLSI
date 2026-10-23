#' Save and load NeuralPLSI model files
#'
#' \code{save_model()} writes the engine's portable text model file (weights,
#' settings and index orientation, with exact float32 round trips), the format
#' that Python's \code{NeuralPLSI.save()} and \code{NeuralPLSI.load()} and the
#' C++ API (\code{nplsi_save}, \code{nplsi_load}) read and write. A model saved
#' in one language predicts identically in the others.
#'
#' The file holds the prediction model only: column names, training data and
#' bootstrap results are not saved, and a loaded model labels its coefficients
#' \code{x1, ...} and \code{z1, ...}. Use \code{saveRDS()} to keep the whole R
#' object. SplinePLSI fits have no engine model file; use \code{saveRDS()}.
#'
#' @param fit A \code{neuralplsi} fit.
#' @param file Path of the model file, conventionally with extension \code{.nplsi}.
#' @return \code{save_model()}: \code{file}, invisibly. \code{load_model()}: a
#'   \code{neuralplsi} object without training data.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(400), 200, 2)
#' g <- function(u) 5 * (plogis(2 * u) - 0.5)
#' y <- rbinom(200, 1, plogis(g(drop(x %*% c(0.8, 0.6)))))
#' fit <- neuralplsi(x, NULL, y, family = "binary")
#' file <- tempfile(fileext = ".nplsi")
#' save_model(fit, file)
#' restored <- load_model(file)
#' identical(predict(restored, x), predict(fit, x))
#' @export
save_model <- function(fit, file) {
  if (!inherits(fit, "neuralplsi"))
    stop("save_model() writes NeuralPLSI fits; save other fits with saveRDS()", call. = FALSE)
  .Call(C_save, engine(fit), model_path(file))
  invisible(file)
}

#' @rdname save_model
#' @export
load_model <- function(file) neuralplsi_object(.Call(C_load, model_path(file)))

model_path <- function(file) {
  if (!is.character(file) || length(file) != 1L || is.na(file) || !nzchar(file))
    stop("file must be a single path", call. = FALSE)
  path.expand(file)
}
