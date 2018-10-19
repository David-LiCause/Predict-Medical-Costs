
# Import libraries
library(dplyr)
library(nnet)
library(keras)

# Import related R scripts
source("regression_modeling.R")
source("nn_modeling.R")

# Set seed for reproducibility
set.seed(1)

# Import data
data <- read.csv("insurance.csv", header=T, stringsAsFactors = F)

# Convert the factor variables to binary matrices
is_female <- as.numeric(data$sex=='female')
is_smoker <- as.numeric(data$smoker=='yes')
region_bin <- class.ind(as.factor(data$region)) %>% 
  as.data.frame() %>%
  dplyr::select(1:3)
# Combine continuous features and dummy coded factor features into single dataframe
data <- cbind(data[,c(7,1,3,4)], is_female, is_smoker, region_bin)
data$k <- sample(1:5, nrow(data), replace = T)

# Benchmark the performance of different types of models
regression_error <- regression_eval(data, k_dev=1, k_test=2)
sn_nn_error <- nn_eval(data, k_dev=1, k_test=2, model='single_nueron')
shallow_nn_error <- nn_eval(data, k_dev=1, k_test=2, model='shallow_nueral_network')
deep_nn_error <- nn_eval(data, k_dev=1, k_test=2, model='deep_nueral_network')

# Examine results
results <- as.data.frame(matrix(NA, 4, 4))
colnames(results) <- c("Method", "RMSE Train", "RMSE Dev", "RMSE Test")
results[,1] <- c("Multivariate linear regression", "Single Neuron Neural Network", "Shallow Neural Network", "Deep Neural Network")
results[1,c(2:4)] <- c(unlist(regression_error))
results[2,c(2:4)] <- c(unlist(sn_nn_error))
results[3,c(2:4)] <- c(unlist(shallow_nn_error))
results[4,c(2:4)] <- c(unlist(deep_nn_error))
results 

