import os
import numpy as np
from ANN import ANN
from ProblemSpace import ProblemSpace
from ActivationFunctions import *
from preprocessing import get_preprocessed_data  # Import the preprocessed data
from ActivationFunctions import logistic_regression, hyperbolic_tangent, ReLU
from datetime import datetime  # To keep test outputs organised

def mean_absolute_error(y_true, y_pred):
    # Ensure both inputs are Numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Flatten the arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    # Calculate the MAE
    return np.mean(np.abs(y_true - y_pred))

if __name__ == "__main__":
    

    X_train, y_train, X_test, y_test = get_preprocessed_data()

    ann = ANN(input_size=X_train.shape[1])

    # Add layers
    ann.add_hidden_layer(size=5, activation_function=ReLU)  # First hidden layer
    ann.add_hidden_layer(
        size=4, activation_function=hyperbolic_tangent
    )  # Second hidden layer
    ann.add_hidden_layer(
        size=1, activation_function=logistic_regression
    )  # Output layer

    iters = 10

    # Create problemspace 
    pso = ProblemSpace(X_train, y_train, ann, 10)
    pso.k_iters(iters)

    ann.updateParameters(pso.get_best_location())

    print(f'Testing trained ANN...')

    y_pred_train = ann.forward_pass(X_train)
    mae_train_for = mean_absolute_error(y_train, y_pred_train)

    y_pred_test = ann.forward_pass(X_test)
    mae_test_for = mean_absolute_error(y_test, y_pred_test)

    print(f"Mean Absolute Error (Training): {mae_train_for:.4f}\n")
    print(f"Mean Absolute Error (Test): {mae_test_for:.4f}\n")