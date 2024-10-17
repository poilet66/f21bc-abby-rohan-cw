from main import Perceptron, ReLU

# I wanted to put this inside a /tests/ folder but that made importing from main weird,
# think it may be worth setting up modules properly though

"""
=============================================================
    Testing Getting/Setting of Perceptron Weights + Bias
=============================================================
"""

perceptron = Perceptron(3, ReLU)
print(f"Perceptron weights: {perceptron.weights}\nPerceptron bias: {perceptron.bias}")