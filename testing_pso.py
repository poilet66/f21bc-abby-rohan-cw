import numpy as np
from objects.ANN import ANN
from ProblemSpace import ProblemSpace
from objects.ActivationFunctions import logistic_regression, hyperbolic_tangent, ReLU
from preprocessing import get_preprocessed_data


def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))

def fitness_function(params):
        ann.updateParameters(params)
        train_pred = ann.forward_pass(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        return -train_mae  # Negative since we're maximising

def test_iterations(epochs, iterations, problem_space: ProblemSpace, bounds):
    X_train, y_train, X_test, y_test = get_preprocessed_data()

    test_maes = []
    train_maes = []

    for i in range(iterations):
        iteration_ps: ProblemSpace = problem_space.copy()

        iteration_ps.optimise(
            epochs=epochs,
            bounds=bounds,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        test_maes.append(iteration_ps.best_test_mae)
        train_maes.append(iteration_ps.best_training_mae)
        
    average_test_mae = np.mean(test_maes)
    average_train_mae = np.mean(train_maes)
    std_test_mae = np.std(test_maes)
    std_train_mae = np.std(train_maes)

    return (average_train_mae, average_test_mae, std_train_mae, std_test_mae)

if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test = get_preprocessed_data()

    # build our ann !
    ann = ANN(input_size=X_train.shape[1])
    ann.add_hidden_layer(size=8, activation_function=logistic_regression)
    ann.add_hidden_layer(size=8, activation_function=hyperbolic_tangent)
    ann.add_hidden_layer(size=4, activation_function=ReLU)
    ann.add_hidden_layer(size=2, activation_function=ReLU)
    ann.add_hidden_layer(size=1, activation_function=ReLU)  # Linear output

    # define pso hyperparameters!
    bounds = (-5, 5)
    num_particles = 50

    # Create ProblemSpace with more particles
    pso = ProblemSpace(
        ann=ann,
        num_particles=num_particles,
        num_dimensions=ann.countParams(),
        bounds=bounds,
        fitness_function=fitness_function,
    )

    ret = test_iterations(100, 10, pso, bounds)

    print(ret)
