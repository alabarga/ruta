#' Create an autoencoder learner
#'
#' @param layers Layer construct of class \code{"rutaNetwork"}
#' @param sparse Logical
#' @param contractive Logical
#'
#' @export
autoencoder <- function(layers, sparse = F, contractive = F) {
  learner = list(
    layers = layers,
    sparse = sparse,
    contractive = contractive
  )
  class(learner) = rutaLearner
  learner
}

#' @import kerasR
#' @importFrom reticulate tuple
#' @importFrom reticulate import
makeAutoencoder <- function(learner, input_shape) {
  loadKeras()

  input_shape = tuple(as.integer(input_shape))
  encoding_dim = as.integer(32)
  layers = import("keras.layers")
  models = import("keras.models")

  input_img = layers$Input(shape = input_shape)
  encoded = layers$Dense(encoding_dim, activation = "tanh")(input_img)
  decoded = layers$Dense(input_shape[[0]], activation = "sigmoid")(encoded)

  model = models$Model(input_img, decoded)
  encoder = models$Model(input_img, encoded)

  encoded_input = layers$Input(shape = tuple(encoding_dim))
  decoder_layer = model$layers[[3]]

  ### needs adapting for multiple hidden layers
  decoder = models$Model(encoded_input, decoder_layer(encoded_input))

  kerasR::plot_model(model)

  autoencoder_obj = list(
    learner = learner,
    model = model,
    encoder = encoder,
    decoder = decoder
  )
  class(autoencoder_obj) = rutaAutoencoder
  autoencoder_obj
}

#' @rdname train.rutaLearner
#' @export
train <- function(learner, ...)
  UseMethod("train")

#' Train a learner object with data
#'
#' This function compiles the neural network described by the learner object
#' and trains it with the input data.
#' @param learner A \code{"rutaLearner"} object
#' @param data Training data: columns are attributes and rows are instances
#' @param validation_data Additional data.frame of data which will not be used
#' for training but the loss measure will be calculated against it
#' @param epochs The number of times data will pass through the network
#' @param ... Additional parameters for \code{keras_fit}
#' @importFrom kerasR keras_compile
#' @importFrom kerasR keras_fit
#' @export
train.rutaLearner <- function(learner, data, validation_data = NULL, epochs = 100, ...) {
  ae <- makeAutoencoder(learner, input_shape = ncol(data))

  keras_compile(ae$model, optimizer = RMSprop(), loss = "binary_crossentropy")
  keras_fit(
    ae$model,
    x = data,
    y = data,
    batch_size = 256,
    epochs = epochs,
    ...
  )

  ae
}

#' Retrieve encoding of data
#'
#' Extracts the encoding calculated by a trained autoencoder for the specified
#' data.
#' @param ae Autoencoder model
#' @param data data.frame to be encoded
#' @export
encode <- function(ae, data) {
  ae$encoder$predict(data)
}

