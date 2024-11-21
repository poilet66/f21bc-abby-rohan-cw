import numpy as np


class Particle:
    def __init__(self, num_dimensions: int, bounds: tuple):
        """
        Initialise a particle with random position and velocity.
        """
        self.location = np.random.uniform(bounds[0], bounds[1], size=num_dimensions)
        velocity_range = (bounds[1] - bounds[0]) * 0.1
        self.velocity = np.random.uniform(
            -velocity_range, velocity_range, size=num_dimensions
        )
        self.best_location = self.location.copy()
        self.best_fitness = float("-inf")
        self.stagnant_counter = 0  # Track iterations without improvement

    def update_velocity(self, global_best, w, c1, c2, bounds, epoch, total_epochs):
        """
        Update velocity with local minima escape mechanism.
        """
        # Random perturbation if stuck in local minimum
        if self.stagnant_counter > 10:
            perturbation = np.random.normal(0, 0.1, size=len(self.location))
            self.velocity += perturbation
            self.stagnant_counter = 0

        cognitive = c1 * np.random.random() * (self.best_location - self.location)
        social = c2 * np.random.random() * (global_best - self.location)

        # Add momentum with adaptive weight
        momentum = w * (1 - epoch / total_epochs) * self.velocity
        self.velocity = momentum + cognitive + social

        # Dynamic velocity clamping with more aggressive early exploration
        max_velocity = (bounds[1] - bounds[0]) * np.exp(-2 * epoch / total_epochs)
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def update_location(self, bounds: tuple):
        """
        Update the particle's location and handle boundary violations.
        """
        # Update position
        self.location += self.velocity

        # Boundary handling - clip to bounds
        self.location = np.clip(self.location, bounds[0], bounds[1])

        # If particle hits boundary, reflect velocity
        hit_min = self.location <= bounds[0]
        hit_max = self.location >= bounds[1]
        self.velocity[hit_min | hit_max] *= -0.5  # Reflect with damping

    def evaluate_fitness(self, fitness_function: callable):
        """
        Evaluate fitness with stagnation detection.
        """
        fitness = fitness_function(self.location)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_location = self.location.copy()
            self.stagnant_counter = 0
        else:
            self.stagnant_counter += 1
