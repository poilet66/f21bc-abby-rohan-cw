import numpy as np

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
        self.weights = np.random.rand(input_size, 1)  # Shape: (input_size, 1)
        # Each perceptron has one weight per input feature. So, the weights array has one row per input feature,
        # and only one column (one weight per feature).

        self.bias = np.random.rand(1)  # Shape: (1,)
        # The bias is a single value added to the weighted sum of inputs before applying the activation function.

        # Store the activation function
        self.activation_function = activation_function

    def update(self, new_weights: np.ndarray, new_bias: float):
        self.weights = new_weights
        self.bias = new_bias

    def numParams(self) -> int:
        return len(self.weights) + 1 # All weights + bias
    
    def updateParams(self, params: np.ndarray) -> None:
        input_size = self.weights.shape[0]
        self.weights = params[:input_size].reshape((input_size, 1)) # update weight (make sure array is of length input_size + is in correct shape)
        self.bias = params[input_size] # bias is last input number

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
