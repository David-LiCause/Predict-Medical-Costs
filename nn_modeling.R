
# Replicate linear regression with nueral network with a single neuron
sn_nn_model <- function(train_x, train_y, dev_x, dev_y, epochs=2500, batch_size=512) {
  # Instantiate keras neural net model
  model <- keras_model_sequential() 
  # Define CNN model architecture
  model %>%
    layer_dense(units = 1, input_shape = c(8), kernel_initializer='he_normal')
  # Compile model
  model %>% compile(
    loss = 'mse',
    # In order to replicate the linear regression model, the single neuron NN must have a very high learning rate
    optimizer = optimizer_adam(lr = 5)) 
  # Fit model using training data
  history <- model %>% fit(
    train_x, train_y,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(dev_x, dev_y),
    shuffle = TRUE
  )
  return(list(model = model,
              history = history))
}

# Evaluate performance of nueral network using root mean squared error
nn_evaluate <- function(model, eval_x, eval_y) {
  mse <- model %>% evaluate(eval_x, eval_y)
  error <- sqrt(mse)
  return(error)
}

# Generate shallow NN model with 4 layers
shallow_nn_model <- function(train_x, train_y, dev_x, dev_y, epochs=2500, batch_size=512) {
  # Instantiate keras neural net model
  model <- keras_model_sequential() 
  # Define CNN model architecture
  model %>%
    layer_dense(units = 32, activation = 'relu', input_shape = c(8), kernel_initializer='he_normal') %>% 
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dropout(rate = .3) %>%
    layer_dense(units = 1, kernel_initializer='he_normal')
  # Compile model
  model %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(lr = .001)) 
  # Fit model using training data
  history <- model %>% fit(
    train_x, train_y,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(dev_x, dev_y),
    shuffle = TRUE
  )
  return(list(model = model,
              history = history))
}

# Generate deep NN model with 9 layers
deep_nn_model <- function(train_x, train_y, dev_x, dev_y, epochs=2500, batch_size=512) {
  # Instantiate keras neural net model
  model <- keras_model_sequential() 
  # Define CNN model architecture
  model %>%
    layer_dense(units = 32, activation = 'relu', input_shape = c(8), kernel_initializer='he_normal') %>% 
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dense(units = 32, activation = 'relu', kernel_initializer='he_normal') %>%
    layer_dropout(rate = .5) %>%
    layer_dense(units = 1, kernel_initializer='he_normal')
  # Compile model
  model %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(lr = .001)) 
  # Fit model using training data
  history <- model %>% fit(
    train_x, train_y,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(dev_x, dev_y),
    shuffle = TRUE
  )
  return(list(model = model,
              history = history))
}

# Evaluate performance of different NN architectures
nn_eval <- function(data, k_dev, k_test, model) {
  # Split data into train, test, dev sets based on k column
  sp_data <- split_data(data, k_dev, k_test)
  train_x <- sp_data$train[,c(2:9)] %>% as.matrix()
  train_y <- sp_data$train[,c(1)] %>% as.matrix()
  dev_x <- sp_data$dev[,c(2:9)] %>% as.matrix()
  dev_y <- sp_data$dev[,c(1)] %>% as.matrix()
  test_x <- sp_data$test[,c(2:9)] %>% as.matrix()
  test_y <- sp_data$test[,c(1)] %>% as.matrix()
  # Generate NN model
  if (model == 'single_nueron') {
    model <- sn_nn_model(train_x, train_y, dev_x, dev_y, epochs=2500, batch_size=512)
  } else if (model == 'shallow_nueral_network') {
    model <- shallow_nn_model(train_x, train_y, dev_x, dev_y, epochs=2500, batch_size=512)
  } else if (model == 'deep_nueral_network') {
    model <- deep_nn_model(train_x, train_y, dev_x, dev_y, epochs=2500, batch_size=512)
  }
  # Calcuate the error on the train, dev, test sets
  train_rmse <- nn_evaluate(model=model$model, eval_x=train_x, eval_y=train_y)
  dev_rmse <- nn_evaluate(model=model$model, eval_x=dev_x, eval_y=dev_y)
  test_rmse <- nn_evaluate(model=model$model, eval_x=test_x, eval_y=test_y)
  return(list(train_rmse = train_rmse, 
              dev_rmse = dev_rmse, 
              test_rmse = test_rmse))
}

