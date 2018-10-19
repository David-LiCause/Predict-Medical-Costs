
# Split data into train, dev, and test sets
split_data <- function(data, k_dev, k_test) {
  train <- data %>%
    filter(k != k_dev, k != k_test) %>%
    select(-k)
  dev <- data %>%
    filter(k == k_dev) %>%
    select(-k)
  test <- data %>%
    filter(k == k_test) %>%
    select(-k) 
  return(list(train = train,
              dev = dev,
              test = test))
}

# Calculate the root mean squared error for the regression model predictions
calculate_rmse <- function(y, yhat) {
  rmse <- sqrt(mean((yhat - y)^2))
  return(rmse)
}

# Generate multivariate linear regression model
reg_model <- function(train_x, train_y) {
  reg_data <- cbind(train_y, train_x)
  colnames(reg_data)[1] <- 'charges'
  mod <- lm(charges ~ ., data=reg_data)
  return(mod)
}

# Calculate RMSE for regression model
reg_evaluate <- function(reg_mod, eval_x, eval_y) {
  pred <- predict(reg_mod, newdata = eval_x)
  error <- calculate_rmse(eval_y, pred)
  return(error)
}

# Evaluate the performance of the regression model
regression_eval <- function(data, k_dev, k_test) {
  sp_data <- split_data(data, k_dev, k_test)
  train_x <- sp_data$train[,c(2:9)]
  train_y <- sp_data$train[,c(1)]
  dev_x <- sp_data$dev[,c(2:9)]
  dev_y <- sp_data$dev[,c(1)]
  test_x <- sp_data$test[,c(2:9)]
  test_y <- sp_data$test[,c(1)]
  model <- reg_model(train_x, train_y)
  train_rmse <- reg_evaluate(reg_mod=model, eval_x=train_x, eval_y=train_y)
  dev_rmse <- reg_evaluate(reg_mod=model, eval_x=dev_x, eval_y=dev_y)
  test_rmse <- reg_evaluate(reg_mod=model, eval_x=test_x, eval_y=test_y)
  return(list(train_rmse = train_rmse, 
              dev_rmse = dev_rmse, 
              test_rmse = test_rmse))
}

