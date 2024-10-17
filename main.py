import os
import numpy as np
from functools import reduce
from preprocessing import get_preprocessed_data  # Import the preprocessed data
from datetime import datetime  # To keep test outputs organised
from typing import List, Callable, Any

"""
========================================
            Activation Functions
========================================
"""


def logistic_regression(x):
    """Logistic regression activation function."""
    return 1 / (1 + np.exp(-x))


def hyperbolic_tangent(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)


def ReLU(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)


"""
========================================
             Perceptron Class
========================================
"""


class Perceptron:
    def __init__(self, input_size, activation_function):
        """
        Initialise the perceptron with random weights and bias.

        Args:
            input_size (int): Number of input features (number of columns in the input data).
            activation_function (callable): Activation function to use.
        """
        # Initialise weights and bias
        self.weights: np.ndarray[Any, np.dtype[np.float64]] = np.random.rand(input_size, 1)  # Shape: (input_size, 1)
        # Each perceptron has one weight per input feature. So, the weights array has one row per input feature,
        # and only one column (one weight per feature).

        self.bias: int = np.random.rand(1)  # Shape: (1,)
        # The bias is a single value added to the weighted sum of inputs before applying the activation function.

        # Store the activation function
        self.activation_function: Callable = activation_function

    def __str__(self):
        return f"PERCEPTRON{{weights={self.weights.flatten()},bias={self.bias},function={self.activation_function.__name__}}}"

    def output(self, inputs):
        """
        Compute the output of the perceptron for given inputs.

        Args:
            inputs (ndarray): Input data of shape (batch_size, input_size).
            - `batch_size` is the number of examples in the input data (number of rows).
            - `input_size` is the number of features for each example (number of columns).

        Returns:
            ndarray: Output after applying the activation function, shape (batch_size, 1).
            - Output has one value per example in the batch (one prediction per input sample).
        """
        # Inputs shape: (batch_size, input_size)
        # Weights shape: (input_size, 1)
        # Bias shape: (1,)

        # Calculate the weighted sum (sigma)
        sigma = np.dot(inputs, self.weights) + self.bias  # Shape: (batch_size, 1)
        # The dot product between inputs (batch_size, input_size) and weights (input_size, 1)
        # results in (batch_size, 1), giving one weighted sum per example.

        # Apply the activation function
        return self.activation_function(sigma)  # Shape: (batch_size, 1)
        # The activation function is applied to the weighted sum for each example in the batch.

"""
========================================
               ANN Class
========================================
"""


class ANN:
    def __init__(self, input_size):
        """
        Initialise the neural network.

        Args:
            input_size (int): Number of input features.
        """
        self.input_size: int = input_size
        self.layers: List[List[Perceptron]]  = []  # Stores layers of perceptrons

    def add_hidden_layer(self, size, activation_function):
        """
        Add a hidden layer to the network.

        Args:
            size (int): Number of perceptrons in the layer.
            activation_function (callable): Activation function for the layer.
        """
        # Determine the input size for the layer
        if not self.layers:
            layer_input_size = self.input_size  # First layer uses input data size
        else:
            layer_input_size = len(
                self.layers[-1]
            )  # Other layers use size of the previous layer

        # Create the layer with the specified activation function
        layer = [Perceptron(layer_input_size, activation_function) for _ in range(size)]
        # Each perceptron in the layer will take the outputs of the previous layer (or input features) as inputs.
        self.layers.append(layer)

    def forward_for(self, inputs):
        """
        Perform forward propagation using a for loop.

        Args:
            inputs (ndarray): Input data of shape (batch_size, input_size).
            - `batch_size`: The number of examples in the batch.
            - `input_size`: The number of features per example.

        Returns:
            ndarray: Output of the network, shape (batch_size, output_size).
            - `output_size`: Number of perceptrons in the output layer.
        """
        inputs = np.array(inputs)  # Ensure inputs are a NumPy array

        # Forward propagation through each layer
        for layer in self.layers:
            outputs = []
            for perceptron in layer:
                # Each perceptron processes the entire batch
                output = perceptron.output(inputs)  # Shape: (batch_size, 1)
                # Each perceptron returns one output per input example.
                outputs.append(output)
            # Stack outputs horizontally to form inputs for the next layer
            inputs = np.hstack(outputs)  # Shape: (batch_size, layer_size)
            # The outputs from all perceptrons in the current layer are combined into the input for the next layer.

        return inputs  # Final output shape: (batch_size, output_size)

    def forward_reduce(self, inputs):
        """
        Perform forward propagation using the reduce function.

        Args:
            inputs (ndarray): Input data of shape (batch_size, input_size).
            - `batch_size`: The number of examples in the batch.
            - `input_size`: The number of features per example.

        Returns:
            ndarray: Output of the network, shape (batch_size, output_size).
            - `output_size`: Number of perceptrons in the output layer.
        """
        inputs = np.array(inputs)  # Ensure inputs are a NumPy array

        def apply_layer(input_data, layer):
            outputs = [perceptron.output(input_data) for perceptron in layer]
            return np.hstack(outputs)

        return reduce(apply_layer, self.layers, inputs)


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
    output = ann.forward_for(inputs)  # Forward pass using the for-loop method
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
    y_pred_train_for = ann.forward_for(X_train)
    mae_train_for = mean_absolute_error(y_train, y_pred_train_for)

    # Perform forward propagation on the training data using the reduce method
    y_pred_train_reduce = ann.forward_reduce(X_train)
    mae_train_reduce = mean_absolute_error(y_train, y_pred_train_reduce)

    # Perform forward propagation on the test data using the for loop method
    y_pred_test_for = ann.forward_for(X_test)
    mae_test_for = mean_absolute_error(y_test, y_pred_test_for)

    # Perform forward propagation on the test data using the reduce method
    y_pred_test_reduce = ann.forward_reduce(X_test)
    mae_test_reduce = mean_absolute_error(y_test, y_pred_test_reduce)

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
                f"True: {y_train[i]:.4f}, Predicted: {y_pred_train_for[i][0]:.4f}\n"
            )

        # Write training results using reduce
        f.write("\n--- Training Results (Reduce) ---\n")
        f.write(f"Mean Absolute Error (Training): {mae_train_reduce:.4f}\n")
        f.write("First 5 Training Predictions:\n")
        for i in range(5):
            f.write(
                f"True: {y_train[i]:.4f}, Predicted: {y_pred_train_reduce[i][0]:.4f}\n"
            )

        # Write testing results using for loop
        f.write("\n--- Test Results (For Loop) ---\n")
        f.write(f"Mean Absolute Error (Test): {mae_test_for:.4f}\n")
        f.write("First 5 Test Predictions:\n")
        for i in range(5):
            f.write(f"True: {y_test[i]:.4f}, Predicted: {y_pred_test_for[i][0]:.4f}\n")

        # Write testing results using reduce
        f.write("\n--- Test Results (Reduce) ---\n")
        f.write(f"Mean Absolute Error (Test): {mae_test_reduce:.4f}\n")
        f.write("First 5 Test Predictions:\n")
        for i in range(5):
            f.write(
                f"True: {y_test[i]:.4f}, Predicted: {y_pred_test_reduce[i][0]:.4f}\n"
            )

        # Close the file
        f.write("\n===== End of Results =====\n")

    print(f"Results written to {file_name}.")
