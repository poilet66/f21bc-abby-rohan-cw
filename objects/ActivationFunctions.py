import numpy as np

"""
========================================
            Activation Functions
========================================
"""


def logistic_regression(x):
    """Logistic regression activation function."""
    return 1 / (1 + np.exp(-x))


def hyperbolic_tangent(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)


def ReLU(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)
