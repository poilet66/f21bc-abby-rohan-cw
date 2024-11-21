import os
import numpy as np
from objects.ANN import ANN
from objects.ActivationFunctions import logistic_regression, hyperbolic_tangent, ReLU
from preprocessing import get_preprocessed_data
from ProblemSpace import ProblemSpace


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test = get_preprocessed_data()

    # Initialise ANN
    ann = ANN(input_size=X_train.shape[1])
    ann.add_hidden_layer(size=5, activation_function=ReLU)
    ann.add_hidden_layer(size=4, activation_function=hyperbolic_tangent)
    ann.add_hidden_layer(size=1, activation_function=logistic_regression)

    # Define PSO problem space
    def fitness_function(params):
        ann.updateParameters(params)
        predictions = ann.forward_pass(X_train)
        return -mean_absolute_error(y_train, predictions)  # Negative for maximisation

    bounds = (-1, 1)
    problem_space = ProblemSpace(
        ann=ann,
        num_particles=10,
        num_dimensions=ann.countParams(),
        bounds=bounds,
        fitness_function=fitness_function,
    )

    # Optimise ANN using PSO
    problem_space.optimise(iterations=10, bounds=bounds)

    # Evaluate performance
    ann.updateParameters(problem_space.global_best_location)
    y_pred_test = ann.forward_pass(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Mean Absolute Error on Test Set: {mae:.4f}")
