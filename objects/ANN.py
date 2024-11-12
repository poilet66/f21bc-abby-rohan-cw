import numpy as np
from .Perceptron import Perceptron
from functools import reduce
from typing import List, Any

"""
========================================
               ANN Class
========================================
"""


class ANN:
    """
    ANN class :)
    """

    def __init__(self, input_size):
        """
        Initialise the neural network.

        Args:
            input_size (int): Number of input features.
        """
        self.input_size: int = input_size
        self.layers: List[List[Perceptron]]  = []  # Stores layers of perceptrons

    """
    Get shape of ANN (including inputs)
    I.e.: ANN with 3 inputs + following structure:
    [3 -> 2] 
    will return: [3, 3, 2]

    This will help us regenerate said ANN with new parameters while tuning its parameters
    """
    def shape(self) -> List[int]:
        layer_sizes = [len(layer) for layer in self.layers]
        return [self.input_size] + layer_sizes
    
    def countParams(self) -> int:
        total_params = 0
        for layer in self.layers:
            for perceptron in layer:
                total_params += perceptron.numParams()
        return total_params    

    def updateParameters(self, parameters: np.ndarray) -> None:
        start = 0
        for layer in self.layers:
            for perceptron in layer:
                num_params = perceptron.numParams()
                perceptron.updateParams(parameters[start:start + num_params]) # get the slice of the list that corresponds to this perceptrons weights + bias, update
                start += num_params # move window/slice along

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

    def forward_pass(self, inputs):
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
    
    def calculate_loss(self, y_true, y_pred) -> float:
        #get the actual value
        y_true = np.array(y_true).flatten()  # Shape: (batch_size,)
        #get the predicted value
        y_pred = np.array(y_pred).flatten()  # Shape: (batch_size,)

        #calculate loss using MAE
        loss = np.mean(np.abs(y_true - y_pred))
        #returns the calculated loss
        return loss

    # Calculate score of current area (loss from using current position as ANN parameters)
    # I.e.: Set ANN weights/biases to current position -> do forward pass -> calculate loss
    def calculateFitness(self, parameters: np.ndarray):
        #update parameters based on particle position
        self.updateParameters(parameters)

        #calculate the loss using ANN claculate loss method
        currentLoss = self.calculate_loss(y_train, y_pred)
        currentFitness = 1 / (1 + currentLoss)  # ensures fitness is between 0 and 1 (1 is best)
        return self.currentFitness