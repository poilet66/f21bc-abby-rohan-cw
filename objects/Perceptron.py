import numpy as np

"""
========================================
             Perceptron Class
========================================
"""


class Perceptron:
    def __init__(self, input_size: int, activation_function: callable):
        """
        Initialise the perceptron with random weights and bias.

        Args:
            input_size (int): Number of input features.
            activation_function (callable): Activation function.
        """
        self.weights = np.random.rand(input_size, 1)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function

    def numParams(self) -> int:
        """
        Get the number of parameters (weights + bias).
        """
        return self.weights.size + 1

    def updateParams(self, params: np.ndarray) -> None:
        """
        Update perceptron weights and bias.

        Args:
            params (ndarray): Vector of weights and bias.
        """
        if len(params) != self.weights.size + 1:
            raise ValueError("Invalid number of parameters for the perceptron.")
        self.weights = params[:-1].reshape(self.weights.shape)
        self.bias = params[-1]

    def output(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculate the output of the perceptron.

        Args:
            inputs (ndarray): Input data.

        Returns:
            ndarray: Output after applying the activation function.
        """
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Input shape {inputs.shape} does not match weights shape {self.weights.shape}."
            )
        sigma = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(sigma)
