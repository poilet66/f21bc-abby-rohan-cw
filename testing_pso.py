import numpy as np
from objects.ANN import ANN
from ProblemSpace import ProblemSpace
from objects.ActivationFunctions import logistic_regression, hyperbolic_tangent, ReLU
from preprocessing import get_preprocessed_data


def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))


if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test = get_preprocessed_data()

    # Initialise ANN with smaller architecture
    ann = ANN(input_size=X_train.shape[1])
    ann.add_hidden_layer(size=8, activation_function=logistic_regression)
    ann.add_hidden_layer(size=4, activation_function=hyperbolic_tangent)
    ann.add_hidden_layer(size=3, activation_function=ReLU)
    ann.add_hidden_layer(size=2, activation_function=ReLU)
    ann.add_hidden_layer(size=1, activation_function=lambda x: x)  # Linear output

    # Define fitness function
    def fitness_function(params):
        ann.updateParameters(params)
        train_pred = ann.forward_pass(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        return -train_mae  # Negative since we're maximising

    # Tighter PSO bounds
    bounds = (-5, 5)

    # Create ProblemSpace with more particles
    pso = ProblemSpace(
        ann=ann,
        num_particles=50,
        num_dimensions=ann.countParams(),
        bounds=bounds,
        fitness_function=fitness_function,
    )

    # Run PSO optimisation
    pso.optimise(
        epochs=10000,  # Reduced number of epochs
        bounds=bounds,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
