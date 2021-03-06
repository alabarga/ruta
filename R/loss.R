#' @rdname as_loss
#' @export
as_loss.character <- function(x) {
  structure(
    list(
      name = x
    ),
    class = c(ruta_loss_named, ruta_loss)
  )
}

#' @rdname as_loss
#' @export
as_loss.ruta_loss <- function(x) x

#' Obtain a Keras loss
#'
#' @description Builds the Keras loss function corresponding to a name
#'
#' @param x A \code{"ruta_loss_named"} object
#' @return A function which returns the corresponding loss for given true and
#' predicted values
#' @param ... Rest of parameters, ignored
#'
#' @export
to_keras.ruta_loss_named <- function(x, ...) {
  get_keras_object(x$name, "loss")
}

