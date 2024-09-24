import numpy as np
from abc import ABC, abstractmethod
import math

class Perceptron(ABC):
    # Instantiate with weights, bias
    def __init__(self, weights, bias, input_size):
        self.weights = weights
        self.bias = bias

    # NoArgs constructor
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    # Activation function - we need to implement a few so made this abstract
    @abstractmethod
    def activation_function(self):
        pass 

    # Perform activation function on inputs
    def output(self, inputs):
        # Dot product (Matrix multiplication) (I think)
        sigma = np.dot(self.weights, inputs) + self.bias
        return sigma
    
class PerceptronLogistic(Perceptron):

    #Override activation function
    def activation_function(self, x):
        return 1/(1 + math.exp(-x))
    
class PerceptronReLU(Perceptron):

    #Override activation function
    def activation_function(self, x):
        pass

class PerceptronHyperbolicTangent(Perceptron):

    #Override activation function
    def activation_function(self, x):
        pass


class ANN:
    pass

# Main function
if __name__ == "__main__":
    perc = PerceptronLogistic()