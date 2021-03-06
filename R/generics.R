#' @rdname train.ruta_autoencoder
#' @export
train <- function(learner, ...) {
  UseMethod("train")
}

#' Coercion to ruta_loss
#'
#' Generic function to coerce objects into loss objects.
#' @param x Object to be converted into a loss
#' @return A \code{"ruta_loss"} construct
#' @export
as_loss <- function(x) UseMethod("as_loss")

#' Coercion to ruta_network
#'
#' Generic function to coerce objects into networks.
#' @param x Object to be converted into a network
#' @return A \code{"ruta_network"} construct
#' @export
as_network <- function(x) UseMethod("as_network")

#' Convert a Ruta object onto Keras objects and functions
#'
#' Generic function which uses the Keras API to build objects
#' out of Ruta wrappers
#'
#' @param x Object to be converted
#' @param ... Remaining parameters depending on the method
to_keras <- function(x, ...) UseMethod("to_keras")

#' Generate samples from a generative model
#'
#' @param learner Trained learner object
#' @export
generate <- function(learner, ...) UseMethod("generate")
