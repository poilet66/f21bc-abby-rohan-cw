import numpy as np
from abc import ABC, abstractmethod
import math

'''
ACTIVATION FUNCTIONS
'''

def logistic_regression(x):
    return 1/(1 + math.exp(-x))

def hyperbolic_tangent(x):
    return np.tanh(x)

def RelU(x):
    return max(0,x)
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

class ANN:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_hidden_layer(self, size, activation_function):
        # Determine the input size for the layer
        if not self.layers:
            input_size = self.input_size
        else:
            input_size = len(self.layers[-1])

        # Create the layer with the specified activation function
        layer = [Perceptron(input_size, activation_function) for _ in range(size)]
        self.layers.append(layer)

    # Input is list of ints of size self.input_size
    def forward(self, input):
        # Send each input to each perceptron in first layer
        # Send each output from each perceptron to each perceptron in next layer
        # So on so forth until we reach last input layer
        # Send all final outputs to output perceptron
        # Calculate final output
        pass

'''
======================================
           Example usage
======================================
    # Create ANN with shape 3 -> 3 -> 2 -> 1
    ann = ANN(inputSize=3)
    ann.add_hidden_layer(3, activation_type=logistic_regression)
    ann.add_hidden_layer(2, activation_function=logistic_regression)
    ann.add_output_layer(activation_function=typeActivation)

    ann.output([3, 5, 1])
'''
'''
=======================================
 Example of Insantiation of Perceptron
=======================================

Perceptron(input_size=3, activation_function= lambda x: 1/(1 + math.exp(-x)))
Perceptron(input_size=3, activation_function=RelU)
'''

# ANN(input_size = 3, input_layers=[3, 4], output) 0: 3 nodes, 1: 4 nodes,

# Main function
if __name__ == "__main__":
    # perc = PerceptronLogistic(input_size=3)
    # sigma = perc.output([0.3, 0.5, 0.87])
    # print(perc.activation_function(sigma))

    # Create ANN with shape 3 -> 3 -> 2 -> 1
    ann = ANN(inputSize=3)
    ann.add_hidden_layer(3, activation_type=logistic_regression)
    ann.add_hidden_layer(2, activation_function=logistic_regression)