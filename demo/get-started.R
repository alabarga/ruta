library(magrittr)
library(keras)
library(ruta)

plot_digit <- function(digit, ...) {
  image(array_reshape(1 - digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col=gray((0:255)/255), ...)
}

plot_sample <- function(digits_test, digits_dec, sample) {
  sample_size <- length(sample)
  layout(
    matrix(c(1:sample_size, (sample_size + 1):(2 * sample_size)), byrow = F, nrow = 2)
  )

  for (i in sample) {
    par(mar = c(0,0,0,0) + 1)
    plot_digit(digits_test[i, ])
    plot_digit(digits_dec[i, ])
  }
}

plot_matrix <- function(digits) {
  n <- dim(digits)[1]
  layout(
    matrix(1:n, byrow = F, nrow = sqrt(n))
  )

  for (i in 1:n) {
    par(mar = c(0,0,0,0) + .2)
    plot_digit(digits[i, ])
  }
}

mnist = dataset_mnist()
x_train <- array_reshape(
  mnist$train$x, c(dim(mnist$train$x)[1], 784)
)
x_train <- x_train / 255.0
x_test <- array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test <- x_test / 255.0

#----------------Sparse autoencoder---------------------------------------

# Specifies one hidden layer with 36 units and tanh activation function
network <- input() + dense(36, "selu") + output("sigmoid")
learner <-
  network %>%
  autoencoder_sparse(loss = "binary_crossentropy", weight = 0.05)

model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 32
)
decoded <- model %>% reconstruct(x_test)

plot_sample(x_test, decoded, 1:10)

#----------------Contractive autoencoder---------------------------------------

# Specifies one hidden layer with 36 units and tanh activation function
network <- input() + dense(36, "elu") + output("sigmoid")
learner <-
  network %>%
  autoencoder_contractive(loss = "binary_crossentropy", weight = 2e-4)

model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 128#,
  # validation_data = x_test,
  # metrics = list("binary_accuracy")
)
decoded <- model %>% reconstruct(x_test)

plot_sample(x_test, decoded, 1:10)

#----------------Denoising autoencoder-----------------------------------------

# Noisy filters
x_test_noisy <- apply_filter(noise_gaussian(var = 0.01), x_test)
plot_sample(x_test, x_test_noisy, 1:10)


# Specifies one hidden layer with 36 units and tanh activation function
network <- input() + dense(36, "tanh") + output("sigmoid")
learner <-
  network %>%
  autoencoder_denoising(loss = "binary_crossentropy", noise_type = "ones")

model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 64,
  validation_data = x_test,
  metrics = list("binary_accuracy")
)

x_test_noisy <- apply_filter(noise_ones(), x_test)
decoded <- model %>% reconstruct(x_test_noisy)

plot_sample(x_test_noisy, decoded, 1:10)

#----------------Robust autoencoder-----------------------------------------

# Specifies one hidden layer with 36 units and tanh activation function
network <- input() + dense(36, "elu") + output("sigmoid")
learner <-
  network %>%
  autoencoder_robust() %>%
  add_weight_decay()

model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 64,
  validation_data = x_test,
  metrics = list("binary_accuracy")
)

x_test_noisy <- apply_filter(noise_gaussian(), x_test)
decoded <- model %>% reconstruct(x_test_noisy)

plot_sample(x_test_noisy, decoded, 1:10)

#----------------Other layers---------------------------------------

# Specifies one hidden layer with 36 units and tanh activation function
network <-
  input() +
  dense(256, "relu") + dropout() +
  dense(  6, "relu") +
  dense(256, "relu") + dropout() +
  output("sigmoid")
learner <-
  network %>%
  autoencoder(loss = "binary_crossentropy") #%>%
  # Adds weight decay onto the encoding layer
  add_weight_decay()

model <- train(
  learner,
  x_train,
  epochs = 20,
  optimizer = "rmsprop",
  batch_size = 64,
  validation_data = x_test,
  metrics = list("binary_accuracy")
)
decoded <- model %>% reconstruct(x_test)

plot_sample(x_test, decoded, 1:10)

#---------------------Variational autoencoder----------------------------------

network <-
  input() +
  dense(256, "elu") +
  variational_block(3) +
  dense(256, "elu") +
  output("sigmoid")

learner <- autoencoder_variational(network, loss = "binary_crossentropy")

model <- learner %>% train(x_train, epochs = 50)

# decoded <- model %>% reconstruct(x_test)
#
# plot_sample(x_test, decoded, 1:10)

# sampling
model %>% generate(dimensions = c(2, 3), fixed_values = 0.5) %>% plot_matrix()

library(animation)

par(bg = "white")  # ensure the background color is white
plot(c(), type = "n")

ani.record(reset = T)

for (t in seq(from = 0.001, to = 0.999, length.out = 180)) {
  model %>% generate(dimensions = c(2, 3), from = 0.001, to = 0.999, fixed_values = t) %>% plot_matrix()
  ani.record()
}

oopts = ani.options(interval = 1/30)
ani.replay()

saveHTML(ani.replay(), img.name = "record_plot")

plot_matrix(sampled)

