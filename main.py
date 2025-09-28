from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

# Based on a 7:1:2 split
TRAINING_SAMPLES = 538
VALIDATION_SAMPLES = 76
TEST_SAMPLES = 154

# Functions to prep data for linear regression and ridge regression learning:
def decomposeMatrix(data_set):
    y1_df = data_set['Y1']
    y1_vector = y1_df.to_numpy()
    y2_df = data_set['Y2']
    y2_vector = y2_df.to_numpy()
    a_df = data_set[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    features = a_df.to_numpy()

    return (features, y1_vector, y2_vector)

def addBias(features, sample_size) -> np.ndarray:
    bias = np.ones((sample_size, 1))
    final_matrix = np.concatenate([bias, features], axis=1)
    return final_matrix

# Calculation functions
def predictViaLinReg(bias, weights, features):
    prediction = bias + (np.sum(weights * features))
    return prediction

def calcMeanSqError(bias, weights, features, true_values):
    sq_error = []
    for sample, true_value in zip(features, true_values):
        sq_error.append(np.square(predictViaLinReg(bias, weights, sample) - true_value))
    
    return np.mean(sq_error)

def trainRidgeRegression(a_matrix, lambda_val, y_vector):
    return np.matmul(np.linalg.pinv(np.matmul(a_matrix.T, a_matrix) + (lambda_val * np.identity(a_matrix.shape[1]))), np.matmul(a_matrix.T, y_vector))

def optimizeRidgeRegression(training_matrix, training_y1, training_y2, validation_features, validation_y1, validation_y2):
    best_lambda = 0.0
    current_lambda = 0.0
    all_lambdas = []
    all_avg_error = []
    best_avg_error = np.inf

    while current_lambda <= 10:
        all_lambdas.append(current_lambda)
        ridge_weights_y1 = trainRidgeRegression(training_matrix, current_lambda, training_y1)
        ridge_weights_y2 = trainRidgeRegression(training_matrix, current_lambda, training_y2)
        
        y1_mse = calcMeanSqError(ridge_weights_y1[0], ridge_weights_y1[1:], validation_features, validation_y1)
        y2_mse = calcMeanSqError(ridge_weights_y2[0], ridge_weights_y2[1:], validation_features, validation_y2)
        avg_error = (y1_mse + y2_mse ) / 2
        all_avg_error.append(avg_error)
        if avg_error < best_avg_error:
            best_lambda = current_lambda
            best_avg_error = avg_error
        current_lambda += 0.1

    return best_lambda, all_lambdas, all_avg_error  
        
# Load Dataset
energy_efficiency_data = fetch_ucirepo(id=242)

energy_df = energy_efficiency_data.data.original # type: ignore :: Get the Pandas DataFrame

# Split the data, beginning by sampling 70% of the full dataset for training
training_set = energy_df.sample(frac=0.7, random_state=123)
temp_set = energy_df.drop(training_set.index) # Create a temp set containing the remaing 30% of data
validation_set = temp_set.sample(frac=0.33, random_state=123) # Take 33% of the remaining 30% (so 10% of original data)
test_set = temp_set.drop(validation_set.index) # Create the testing set by dropping the validation set from the temp set

training_features, training_y1, training_y2 = decomposeMatrix(training_set)
training_matrix = addBias(training_features, TRAINING_SAMPLES)

# Calculations:
lin_reg_weight_vector_y1 = np.matmul(np.linalg.pinv(np.matmul(training_matrix.T, training_matrix)), np.matmul(training_matrix.T, training_y1))
lin_reg_weight_vector_y2 = np.matmul(np.linalg.pinv(np.matmul(training_matrix.T, training_matrix)), np.matmul(training_matrix.T, training_y2))

w0_y1 = lin_reg_weight_vector_y1[0]
w0_y2 = lin_reg_weight_vector_y2[0]
weights_y1 = lin_reg_weight_vector_y1[1:]
weights_y2 = lin_reg_weight_vector_y2[1:]

test_features, test_y1, test_y2 = decomposeMatrix(test_set)

print(f'Linear Regression MSE for Y1: {calcMeanSqError(w0_y1, weights_y1, test_features, test_y1)}')
print(f'Linear Regression MSE for Y2: {calcMeanSqError(w0_y2, weights_y2, test_features, test_y2)}')

validation_features, validation_y1, validation_y2 = decomposeMatrix(validation_set)

best_lambda, all_lambdas, avg_errors = optimizeRidgeRegression(training_matrix, training_y1, training_y2, validation_features, validation_y1, validation_y2)
print(avg_errors)

print(f'Final lambda chosen: {best_lambda}')

final_ridge_weights_y1 = trainRidgeRegression(training_matrix, best_lambda, training_y1)
final_ridge_weights_y2 = trainRidgeRegression(training_matrix, best_lambda, training_y2)

print(f'Ridge Regression MSE for Y1: {calcMeanSqError(final_ridge_weights_y1[0], final_ridge_weights_y1[1:], test_features, test_y1)}')
print(f'Ridge Regression MSE for Y1: {calcMeanSqError(final_ridge_weights_y2[0], final_ridge_weights_y2[1:], test_features, test_y2)}')