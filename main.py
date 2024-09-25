import numpy as np
from functools import reduce

"""
======================================
           Activation Functions
======================================
"""


def logistic_regression(x):
    return 1 / (1 + np.exp(-x))


def hyperbolic_tangent(x):
    return np.tanh(x)


def ReLU(x):
    return np.maximum(0, x)


"""
======================================
           Perceptron Class
======================================
"""


class Perceptron:
    def __init__(self, input_size, activation_function):
        # Initialise weights and bias
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        # Store the activation function
        self.activation_function = activation_function

    def output(self, inputs):
        # Calculate the weighted sum
        sigma = np.dot(self.weights, inputs) + self.bias
        # Apply the activation function
        return self.activation_function(sigma)


"""
======================================
           ANN Class
======================================
"""


class ANN:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []  # Store layers of perceptrons

    """
    Add a hidden layer with 'size' perceptrons.
    Each perceptron uses the specified activation function.
    """

    def add_hidden_layer(self, size, activation_function):
        # Determine the input size for the layer
        if not self.layers:
            input_size = self.input_size
        else:
            input_size = len(
                self.layers[-1]
            )  # Input size equals the size of the last layer

        # Create the layer with the specified activation function
        layer = [Perceptron(input_size, activation_function) for _ in range(size)]
        self.layers.append(layer)

    """
    ======================================
                Forward Passes
    ======================================
    """

    # Input is list of floats of size self.input_size
    def forward_for(self, inputs):
        # Forward pass using a for loop through the layers.

        # Convert inputs to a column vector
        inputs = np.array(inputs).reshape(-1, 1)

        # Performs forward propagation through each layer using a for loop
        for layer in self.layers:
            outputs = []
            for perceptron in layer:
                # Calculates output for each perceptron in the layer
                outputs.append(perceptron.output(inputs))
            inputs = np.array(
                outputs
            )  # The outputs of this layer become the inputs for the next

        return inputs

    def forward_reduce(self, inputs):
        # Forward pass using reduce to avoid explicit loops.

        # Converts inputs to a column vector
        inputs = np.array(inputs).reshape(-1, 1)

        # Use reduce to apply the forward propagation across layers without for loop
        def apply_layer(input_data, layer):
            outputs = [perceptron.output(input_data) for perceptron in layer]
            return np.array(outputs)

        # Use reduce to sequentially apply each layer
        return reduce(apply_layer, self.layers, inputs)


"""
======================================
         Evaluation Function
======================================
"""


def mean_absolute_error(y_true, y_pred):
    # Ensure both inputs are Numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Flatten the arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate the MAE
    return np.mean(np.abs(y_true - y_pred))


"""
======================================
           Example usage
======================================
    # Create an ANN with shape 3 -> 3 -> 2 -> 1
    ann = ANN(input_size=3)
    ann.add_hidden_layer(size=3, activation_function=logistic_regression)  # First hidden layer
    ann.add_hidden_layer(size=2, activation_function=logistic_regression)  # Second hidden layer
    ann.add_hidden_layer(size=1, activation_function=logistic_regression)  # Output layer

    # Example input (3 features)
    inputs = [3, 5, 1]
    output = ann.forward_for(inputs)  # Forward pass using the for loop method
    print("Output:", output)

=======================================
 Example of Instantiation of Perceptron
=======================================

# Perceptron with 3 inputs using a custom lambda activation function (logistic regression):
perceptron = Perceptron(input_size=3, activation_function=lambda x: 1 / (1 + math.exp(-x)))

# Perceptron with 3 inputs using ReLU activation function:
perceptron = Perceptron(input_size=3, activation_function=ReLU)
"""

# Main function
if __name__ == "__main__":
    # Create an instance of ANN with 10 input features
    ann = ANN(input_size=10)

    # Add layers
    ann.add_hidden_layer(size=5, activation_function=ReLU)  # First hidden layer
    ann.add_hidden_layer(
        size=4, activation_function=hyperbolic_tangent
    )  # Second hidden layer
    ann.add_hidden_layer(
        size=1, activation_function=logistic_regression
    )  # Output layer

    # Example input (10 features)
    inputs = [0.5, -1.2, 3.3, -4.2, 1.46, 2.13, 1.0, 4.0, -7.0, 2.0]

    # Example true values (targets) for MAE calculation
    y_true = [0.8]  # Example expected output for the given inputs

    # Forward propagation using the for loop method
    y_pred_for = ann.forward_for(inputs)
    print("Predicted output using for loop:", y_pred_for)

    # Calculate MAE
    mae_for = mean_absolute_error(y_true, y_pred_for)
    print("For Loop Mean Absolute Error (MAE):", mae_for)

    # Forward propagation using the reduce method
    y_pred_reduce = ann.forward_reduce(inputs)
    print("Predicted output using reduce:", y_pred_reduce)

    # Calculate MAE
    mae_reduce = mean_absolute_error(y_true, y_pred_reduce)
    print("Reduce Mean Absolute Error (MAE):", mae_reduce)
