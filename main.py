import os
import numpy as np
from objects.ANN import ANN
from objects.ActivationFunctions import *
from preprocessing import get_preprocessed_data  # Import the preprocessed data
from datetime import datetime  # To keep test outputs organised
"""
========================================
           Evaluation Function
========================================
"""


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error between true and predicted values.

    Args:
        y_true: True target values (shape: (batch_size,)).
        y_pred: Predicted values (shape: (batch_size,)).

    Returns:
        float: Mean absolute error.
    """
    y_true = np.array(y_true).flatten()  # Shape: (batch_size,)
    y_pred = np.array(y_pred).flatten()  # Shape: (batch_size,)
    return np.mean(np.abs(y_true - y_pred))


"""
========================================
              Example Usage
========================================
Example of creating an ANN with shape 3 -> 3 -> 2 -> 1:

    ann = ANN(input_size=3)
    ann.add_hidden_layer(size=3, activation_function=logistic_regression)  # First hidden layer
    ann.add_hidden_layer(size=2, activation_function=logistic_regression)  # Second hidden layer
    ann.add_hidden_layer(size=1, activation_function=logistic_regression)  # Output layer

    # Example input (3 features)
    inputs = [3, 5, 1]
    output = ann.forward(inputs)  # Forward pass
    print("Output:", output)

Example of instantiating a Perceptron:

    # Perceptron with 3 inputs using a custom lambda activation function (logistic regression):
    perceptron = Perceptron(input_size=3, activation_function=lambda x: 1 / (1 + np.exp(-x)))

    # Perceptron with 3 inputs using ReLU activation function:
    perceptron = Perceptron(input_size=3, activation_function=ReLU)
"""

# Main function
if __name__ == "__main__":
    # Get the current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get preprocessed data
    X_train, y_train, X_test, y_test = get_preprocessed_data()

    # Create an instance of ANN with the number of input features from the dataset
    ann = ANN(input_size=X_train.shape[1])  # Set input size based on dataset

    # Add layers
    ann.add_hidden_layer(size=5, activation_function=ReLU)  # First hidden layer
    ann.add_hidden_layer(
        size=4, activation_function=hyperbolic_tangent
    )  # Second hidden layer
    ann.add_hidden_layer(
        size=1, activation_function=logistic_regression
    )  # Output layer

    # Perform forward propagation on the training data using the for loop method
    y_pred_train = ann.forward_pass(X_train)
    mae_train_for = mean_absolute_error(y_train, y_pred_train)

    # Perform forward propagation on the test data using the for loop method
    y_pred_test = ann.forward_pass(X_test)
    mae_test_for = mean_absolute_error(y_test, y_pred_test)

    # Create the "test_results" directory if it does not exist
    output_dir = "test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a new filename like results1.txt, results2.txt, etc.
    file_index = 1
    while os.path.exists(f"{output_dir}/results{file_index}.txt"):
        file_index += 1
    file_name = f"{output_dir}/results{file_index}.txt"

    # Write results to the new file
    with open(file_name, "w") as f:
        # Write timestamp
        f.write(f"===== ANN Model Results =====\n")
        f.write(f"Timestamp: {current_time}\n")

        # Write training results using for loop
        f.write("\n--- Training Results (For Loop) ---\n")
        f.write(f"Mean Absolute Error (Training): {mae_train_for:.4f}\n")
        f.write("First 5 Training Predictions:\n")
        for i in range(5):
            f.write(
                f"True: {y_train[i]:.4f}, Predicted: {y_pred_train[i][0]:.4f}\n"
            )

        # Write testing results using for loop
        f.write("\n--- Test Results (For Loop) ---\n")
        f.write(f"Mean Absolute Error (Test): {mae_test_for:.4f}\n")
        f.write("First 5 Test Predictions:\n")
        for i in range(5):
            f.write(f"True: {y_test[i]:.4f}, Predicted: {y_pred_test[i][0]:.4f}\n")

        # Close the file
        f.write("\n===== End of Results =====\n")

    print(f"Results written to {file_name}.")
