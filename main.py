from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

# Functions to prep data for linear regression and ridge regression learning:
def decomposeMatrix(data_set):
    y1_df = data_set['Y1']
    y1_vector = y1_df.to_numpy()
    y2_df = data_set['Y2']
    y2_vector = y2_df.to_numpy()
    a_df = data_set[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    features = a_df.to_numpy()

    return (features, y1_vector, y2_vector)

def addBias(features) -> np.ndarray:
    bias = np.ones((features.shape[0], 1), dtype=features.dtype)
    final_matrix = np.concatenate([bias, features], axis=1)
    return final_matrix

# Calculation functions
def predictViaLinReg(bias, weights, features):
    prediction = bias + np.dot(weights, features)
    print(prediction)
    return prediction

def calcMeanSqError(bias, weights, features, true_values):
    sq_error = []
    for sample, true_value in zip(features, true_values):
        print(true_value)
        sq_error.append(np.square(predictViaLinReg(bias, weights, sample) - true_value))
    return np.mean(sq_error)
        
# Load Dataset
energy_efficiency_data = fetch_ucirepo(id=242)

energy_df = energy_efficiency_data.data.original # type: ignore :: Get the Pandas DataFrame

# Split the data, beginning by sampling 70% of the full dataset for training
training_set = energy_df.sample(frac=0.7, random_state=345345)
temp_set = energy_df.drop(training_set.index) # Create a temp set containing the remaing 30% of data
validation_set = temp_set.sample(frac=0.33, random_state=654168) # Take 33% of the remaining 30% (so 10% of original data)
test_set = temp_set.drop(validation_set.index) # Create the testing set by dropping the validation set from the temp set

features, y1_vector, y2_vector = decomposeMatrix(training_set)
a_final_matrix = addBias(features)

# Calculations:
lin_reg_weight_vector_y1 = np.matmul(np.linalg.inv(np.matmul(a_final_matrix.T, a_final_matrix)), np.matmul(a_final_matrix.T, y1_vector))
lin_reg_weight_vector_y2 = np.matmul(np.linalg.inv(np.matmul(a_final_matrix.T, a_final_matrix)), np.matmul(a_final_matrix.T, y2_vector))

bias_y1 = lin_reg_weight_vector_y1[0]
bias_y2 = lin_reg_weight_vector_y2[0]
weights_y1 = lin_reg_weight_vector_y1[1:]
weights_y2 = lin_reg_weight_vector_y2[1:]

test_features, test_y1, test_y2 = decomposeMatrix(test_set)
print(calcMeanSqError(bias_y1, weights_y1, test_features, test_y1))
print(calcMeanSqError(bias_y2, weights_y2, test_features, test_y2))