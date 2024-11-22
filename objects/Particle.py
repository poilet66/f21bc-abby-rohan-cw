import numpy as np


class Particle:
    def __init__(self, num_dimensions: int, bounds: tuple):
        """
        Initialise a particle with random position and velocity.
        """
        self.location = np.random.uniform(bounds[0], bounds[1], size=num_dimensions) # initialise with a random position within the bounds
        velocity_range = (bounds[1] - bounds[0]) * 0.1 #initialise velocity range with smaller range than the position
        self.velocity = np.random.uniform(
            -velocity_range, velocity_range, size=num_dimensions
        )
        #initialise best known position and fitness
        self.best_location = self.location.copy()
        self.best_fitness = float("-inf")
        self.not_moved_counter = 0  # Track iterations without improvement

    def update_velocity(self, global_best, w, c1, c2, bounds, epoch, total_epochs):
        """
        Update velocity with random movement if stuck
        """
        # Random movement if stuck for 10 epochs
        if self.not_moved_counter > 10:
            movement = np.random.normal(0, 0.1, size=len(self.location))
            self.velocity += movement
            self.not_moved_counter = 0

        cognitive = c1 * np.random.random() * (self.best_location - self.location) #pulls towards particle's best location
        social = c2 * np.random.random() * (global_best - self.location) # pull towards global best location

        # decrease momentum over time
        momentum = w * (1 - epoch / total_epochs) * self.velocity

        self.velocity = momentum + cognitive + social #mix movements together 

        # Dynamic velocity clamping with more aggressive early exploration
        max_velocity = (bounds[1] - bounds[0]) * np.exp(-2 * epoch / total_epochs)
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def update_location(self, bounds: tuple):
        """
        Update the particle's location and handle boundary violations.
        """
        # Update position based on velocity
        self.location += self.velocity

        # Boundary handling - stops particles going out of bounds
        self.location = np.clip(self.location, bounds[0], bounds[1])

        # If particle hits boundary, bounce back slower
        hit_min = self.location <= bounds[0]
        hit_max = self.location >= bounds[1]
        self.velocity[hit_min | hit_max] *= -0.5  # bounce back at half speed

    def evaluate_fitness(self, fitness_function):
        """
        Evaluate fitness with stagnation detection.
        """
        fitness = fitness_function(self.location)

        if fitness > self.best_fitness:
            self.best_fitness = fitness # updates best fitness
            self.best_location = self.location.copy() # remember best
            self.not_moved_counter = 0 # reset no movement counter
        else:
            self.not_moved_counter += 1 
