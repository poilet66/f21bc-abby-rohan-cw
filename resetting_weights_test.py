from objects.Perceptron import Perceptron
from objects.ActivationFunctions import ReLU
from objects.ANN import ANN
import numpy as np

"""
=============================================================
    Testing Getting/Setting of Perceptron Weights + Bias
=============================================================
"""


def test_perceptron():
    perceptron = Perceptron(3, ReLU)
    print(
        f"Perceptron weights: {perceptron.weights}\nPerceptron bias: {perceptron.bias}"
    )

    weights = perceptron.weights + 0.1
    perceptron.updateParams(np.append(weights.flatten(), perceptron.bias + 0.1))

    print(
        f"Updated Perceptron weights: {perceptron.weights}\nUpdated bias: {perceptron.bias}"
    )


def test_ann():
    ann = ANN(3)
    ann.add_hidden_layer(2, ReLU)
    ann.add_hidden_layer(1, ReLU)

    for layer in ann.layers:
        for perceptron in layer:
            perceptron.updateParams(
                np.append(perceptron.weights.flatten() + 0.1, perceptron.bias + 0.1)
            )

    print(ann.layers[0][0])


if __name__ == "__main__":
    test_ann()
