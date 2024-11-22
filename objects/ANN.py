import numpy as np
from .Perceptron import Perceptron
from typing import List, Any

"""
========================================
               ANN Class
========================================
"""


class ANN:
    """
    Implementation of a simple Artificial Neural Network.
    """

    def __init__(self, input_size):
        """
        Initialise the ann.
        """
        self.input_size: int = input_size
        self.layers: List[List[Perceptron]] = [] # list of layers each containing perceptrons

    def shape(self) -> List[int]:
        """
        Get the structure of the ANN, including input size and number of perceptrons in each layer.
        """
        layer_sizes = [len(layer) for layer in self.layers]
        return [self.input_size] + layer_sizes

    def countParams(self) -> int:
        """
        Count the total number of trainable parameters (weights + biases) in the network.
        """
        total_params = 0
        for layer in self.layers:
            for perceptron in layer:
                total_params += perceptron.numParams()
        return total_params

    def updateParameters(self, parameters) -> None:
        """
        Update the parameters of the network using p article location vector.
        """
        start = 0
        for layer in self.layers:
            for perceptron in layer:
                num_params = perceptron.numParams()
                perceptron.updateParams(parameters[start : start + num_params])
                start += num_params

    def add_hidden_layer(self, size, activation_function) -> None:
        """
        Add a hidden layer to ann
        """
        layer_input_size = self.input_size if not self.layers else len(self.layers[-1]) # determin input size of new layer
        layer = [Perceptron(layer_input_size, activation_function) for _ in range(size)] # create layer with specified number of neurons
        self.layers.append(layer)

    def forward_pass(self, inputs) -> np.ndarray:
        """
        forward pass
        """
        inputs = np.array(inputs)

        for layer in self.layers: # process through each perceptron in the layer
            outputs = []
            for perceptron in layer:
                outputs.append(perceptron.output(inputs))

            inputs = np.hstack(outputs) # combine perceptron outputs for next layer
        return inputs

    def calculate_loss(self, y_true, y_pred) -> float:
        """
        Calculate loss of prediction
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return np.mean(np.abs(y_true - y_pred))
