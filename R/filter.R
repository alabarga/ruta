#' Apply filters
#'
#' @description Apply a filter to input data, generally a noise filter in
#' order to train a denoising autoencoder. Users won't generally need to use
#' these functions
#'
#' @param filter Filter object to be applied
#' @param data Input data to be filtered
#' @param ... Other parameters
#'
#' @seealso `\link{autoencoder_denoising}`
#' @export
apply_filter <- function(filter, data, ...) UseMethod("apply_filter")

#' @rdname apply_filter
#' @export
apply_filter.ruta_custom <- function(filter, data, ...) {
  filter(data, ...)
}

runif_matrix <- function(data) {
  dims <- dim(data)
  dims %>%
    prod() %>%
    stats::runif() %>%
    matrix(nrow = dims[1], ncol = dims[2])
}

new_noise <- function(cl, ...) {
  structure(
    list(...),
    class = c(cl, ruta_noise, ruta_filter)
  )
}

#' Noise generator
#'
#' Delegates on noise classes to generate noise of some type
#' @param type Type of noise, as a character string
#' @param ... Parameters for each noise class
#' @export
noise <- function(type, ...) {
  noise_f <- switch(tolower(type),
                    zeros = noise_zeros,
                    ones = noise_ones,
                    saltpepper = noise_saltpepper,
                    gaussian = noise_gaussian,
                    cauchy = noise_cauchy,
                    NULL
  )

  if (is.null(noise_f)) {
    stop("Invalid noise type selected")
  }

  noise_f(...)
}

#' Filter to add zero noise
#'
#' A data filter which replaces some values with zeros
#'
#' @param p Probability that a feature in an instance is set to zero
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_zeros <- function(p = 0.05) {
  new_noise(ruta_noise_zeros, p = p)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_zeros <- function(filter, data, ...) {
  multiplier <- as.integer(runif_matrix(data) > filter$p)
  data * multiplier
}

#' Filter to add ones noise
#'
#' A data filter which replaces some values with ones
#'
#' @param p Probability that a feature in an instance is set to one
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_ones <- function(p = 0.05) {
  new_noise(ruta_noise_ones, p = p)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_ones <- function(filter, data, ...) {
  term <- runif_matrix(data)
  data[term < filter$p] <- 1
  data
}

#' Filter to add salt-and-pepper noise
#'
#' A data filter which replaces some values with zeros or ones
#'
#' @param p Probability that a feature in an instance is set to zero or one
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_saltpepper <- function(p = 0.05) {
  new_noise(ruta_noise_saltpepper, p = p)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_saltpepper <- function(filter, data, ...) {
  saltpepper <- runif_matrix(data)
  zero_mask <- saltpepper < filter$p/2
  one_mask <- saltpepper > (1 - filter$p/2)

  data[zero_mask] <- 0
  data[one_mask] <- 1
  data
}

#' Additive Gaussian noise
#'
#' A data filter which adds Gaussian noise to instances
#'
#' @param sd Standard deviation for the Gaussian distribution
#' @param var Variance of the Gaussian distribution (optional, only used
#'  if `sd` is not provided)
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_gaussian <- function(sd = NULL, var = NULL) {
  if (is.null(sd)) {
    sd <- if (is.null(var))
      0.1
    else
      sqrt(var)
  }

  new_noise(ruta_noise_gaussian, sd = sd)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_gaussian <- function(filter, data, ...) {

  dims <- dim(data)
  term <-
    dims %>%
    prod() %>%
    stats::rnorm(sd = filter$sd) %>%
    matrix(nrow = dims[1], ncol = dims[2])

  data + term
}

#' Additive Cauchy noise
#'
#' A data filter which adds noise from a Cauchy distribution to instances
#'
#' @param scale Scale for the Cauchy distribution
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_cauchy <- function(scale = 0.005) {
  new_noise(ruta_noise_cauchy, scale = scale)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_cauchy <- function(filter, data, ...) {
  dims <- dim(data)
  term <-
    dims %>%
    prod() %>%
    stats::rcauchy(scale = filter$scale) %>%
    matrix(nrow = dims[1], ncol = dims[2])

  data + term
}
