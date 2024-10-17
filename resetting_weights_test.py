from main import Perceptron, ReLU, ANN

# I wanted to put this inside a /tests/ folder but that made importing from main weird,
# think it may be worth setting up modules properly though

"""
=============================================================
    Testing Getting/Setting of Perceptron Weights + Bias
=============================================================
"""

def test_perceptron():

    # Getting
    perceptron = Perceptron(3, ReLU)
    print(f"Perceptron weights: {perceptron.weights}\nPerceptron bias: {perceptron.bias}")

    # Setting
    weights = perceptron.weights
    for weight in weights:
        weight += 0.1
    perceptron.weights = weights

    perceptron.bias += 0.1

    print(f"Updated Perceptron weights: {perceptron.weights}\nPerceptron bias: {perceptron.bias}")

"""
=============================================================
        Testing Getting/Setting of ANN Perceptrons
=============================================================
"""

def test_ann():

    ann = ANN(3)
    ann.add_hidden_layer(2, ReLU)
    ann.add_hidden_layer(1, ReLU)

    # Add 0.1 to every value in each perceptron of each layer of ANN
    for layer in ann.layers:
        for perceptron in layer:
            for weight in perceptron.weights:
                weight += 1
            perceptron.bias += 1
    
    print(ann.layers[0][0])

if __name__ == "__main__":
    test_ann()