% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/autoencoder.R, R/autoencoder_variational.R
\name{to_keras.ruta_autoencoder}
\alias{to_keras.ruta_autoencoder}
\alias{to_keras.ruta_autoencoder_variational}
\title{Extract Keras models from an autoencoder wrapper}
\usage{
\method{to_keras}{ruta_autoencoder}(learner, input_shape,
  encoder_end = "encoding", decoder_start = "encoding")

\method{to_keras}{ruta_autoencoder_variational}(learner, input_shape)
}
\arguments{
\item{learner}{Object of class \code{"ruta_autoencoder"}}

\item{input_shape}{Number of attributes in input data}

\item{encoder_end}{Name of the Keras layer where the encoder ends}

\item{decoder_start}{Name of the Keras layer where the decoder starts}
}
\value{
A list with several Keras models:
\itemize{
\item \code{autoencoder}: model from the input layer to the output layer
\item \code{encoder}: model from the input layer to the encoding layer
\item \code{decoder}: model from the encoding layer to the output layer
}
}
\description{
Extract Keras models from an autoencoder wrapper
}
\seealso{
\code{\link{autoencoder}}
}
