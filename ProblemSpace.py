import numpy as np
import copy
from objects.Particle import Particle


class ProblemSpace:
    def __init__(
        self,
        ann,
        num_particles: int,
        num_dimensions: int,
        bounds: tuple,
        fitness_function: callable,
        w: float = 0.729,
        c1: float = 1.49445,
        c2: float = 1.49445,
    ):
        self.ann = ann
        self.fitness_function = fitness_function
        self.global_best_location = None
        self.global_best_fitness = float("-inf")
        self.best_training_mae = float("inf")
        self.best_test_mae = float("inf")
        self.stagnation_count = 0
        self.best_fitness_history = []

        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Create particles with improved exploration
        self.particles = [
            Particle(num_dimensions, bounds) for _ in range(num_particles)
        ]

        # Initialise with diverse positions
        for i, particle in enumerate(self.particles):
            particle.evaluate_fitness(self.fitness_function)
            if particle.best_fitness > self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_location = particle.best_location.copy()

    def update_global_best(self) -> None:
        """
        Update the global best position and fitness based on all particles.
        """
        for particle in self.particles:
            if particle.best_fitness > self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_location = particle.best_location.copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

    def optimise(self, epochs: int, bounds: tuple, X_train, y_train, X_test, y_test):
        """
        Optimised PSO with improved exploration and stagnation handling.
        """
        epochs_without_improvement = 0
        best_solution = None

        for epoch in range(epochs):
            # Adaptive parameters based on progress
            progress = epoch / epochs
            self.w = 0.729 * (1 - 0.5 * progress)
            self.c1 = 1.49445 * (
                1 - 0.3 * progress
            )  # Reduce cognitive component over time
            self.c2 = 1.49445 * (
                1 + 0.3 * progress
            )  # Increase social component over time

            # Track changes
            max_position_change = 0
            max_velocity_change = 0

            for particle in self.particles:
                previous_location = particle.location.copy()
                previous_velocity = particle.velocity.copy()

                particle.update_velocity(
                    global_best=self.global_best_location,
                    w=self.w,
                    c1=self.c1,
                    c2=self.c2,
                    bounds=bounds,
                    epoch=epoch,
                    total_epochs=epochs,
                )

                particle.update_location(bounds)
                particle.evaluate_fitness(self.fitness_function)

                max_position_change = max(
                    max_position_change,
                    np.max(np.abs(particle.location - previous_location)),
                )
                max_velocity_change = max(
                    max_velocity_change,
                    np.max(np.abs(particle.velocity - previous_velocity)),
                )

            # Update global best
            previous_best = self.global_best_fitness
            self.update_global_best()

            # Apply global best to ANN
            self.ann.updateParameters(self.global_best_location)

            # Evaluate performance
            y_train_pred = self.ann.forward_pass(X_train)
            train_mae = np.mean(np.abs(y_train - y_train_pred.flatten()))

            y_test_pred = self.ann.forward_pass(X_test)
            test_mae = np.mean(np.abs(y_test - y_test_pred.flatten()))

            # Early stopping with patience
            if test_mae < self.best_test_mae:
                self.best_test_mae = test_mae
                best_solution = self.global_best_location.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Reassign training mae if needed
            self.best_training_mae = min(self.best_training_mae, train_mae)

            # Random restart if stuck
            if epochs_without_improvement >= 30:
                print(f"Triggering random restart at epoch {epoch + 1}")
                for particle in self.particles:
                    if (
                        np.random.random() < 0.5
                    ):  # Randomly reinitialise 50% of particles
                        particle.location = np.random.uniform(
                            bounds[0], bounds[1], size=len(particle.location)
                        )
                        particle.velocity = np.random.uniform(
                            -(bounds[1] - bounds[0]) * 0.1,
                            (bounds[1] - bounds[0]) * 0.1,
                            size=len(particle.location),
                        )
                epochs_without_improvement = 0

            # Stop if truly converged
            if epochs_without_improvement >= 50:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement")
                break

            print(
                f"Epoch {epoch + 1}/{epochs}, Training MAE: {train_mae:.4f}, "
                f"Test MAE: {test_mae:.4f}, Best Test MAE: {self.best_test_mae:.4f}, "
                f"Learning Rate: {self.w:.4f}"
            )

        # Restore best solution found
        if best_solution is not None:
            self.ann.updateParameters(best_solution)
            self.global_best_location = best_solution

    def copy(self):
        """
        Creates a deep copy of the ProblemSpace instance.
        Returns a new ProblemSpace object with copies of all attributes.
        """
        # Use copy.deepcopy to create a completely new instance
        return copy.deepcopy(self)
    
