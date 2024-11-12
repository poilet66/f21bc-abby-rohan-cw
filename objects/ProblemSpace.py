from Particle import Particle
from main import ANN, Perceptron
from typing import List
import numpy as np
from preprocessing import get_preprocessed_data

x_train, y_train, x_test, y_test = get_preprocessed_data()

class ProblemSpace():
    """
    Encapsulating class for particles + ANN for problem
    May contain methods for calcuating local/global bests, etc.
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, ann: ANN, num_particles: int, # Specific to problem
                 alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.5, delta: float = 0.5, epsilon: float = 0.5 # Hyperparameter option TODO: Do these need to sum to 1?
                 ) -> None:
        
        # Init variables
        self.ann: ANN = ann
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.particles: List[Particle] = []

        ## add our particles
        for i in range(num_particles): self.addParticle(Particle())

        # Assume global best to be first particles loc
        self.global_best = self.particles[0].location
        # set all particles informants, and find their best informant location
        for particle in self.particles:
            particle.informants = particle.getNewInformants()
            particle.previous_fittest_informant_location = particle.fittestInformantLocation()

        # instantiate all particles (current best location etc, current informants)

        # for range in epochs:
        #   do epoch
        # get best location seen (global best)

    def calculate_fitness(self, position: np.ndarray) -> np.float64:
        self.ann.updateParameters(position) # Update ann
        y_pred = self.ann.forward_pass(self.X_train) # Get predictions

        # means squared error
        mse = np.mean((self.y_train - y_pred) ** 2) # TODO: Change this later to allow for different error functions

        # invert and map [0-1]
        return 1 / (1 + mse)

    def addParticle(self, particle: Particle):
        self.particles.append(particle)

    def doEpoch(self) -> None:

        for particle in self.particles:
            # Update pos
            particle.updatePosition()
            # Reassign global best if needed
            if self.calculate_fitness(particle.location) > self.calculate_fitness(self.global_best): 
                self.global_best = particle.location

        # After all particles have moved, update their fields
        for particle in self.particles:
            particle.update_fields()
        
        raise Exception('Not implemented yet')
    
    def get_best_location(self, epochs: int) -> np.ndarray:
        # Do epochs
        for i in range(epochs):
            print(f'Beginning epoch {i}')
            self.doEpoch()
            print(f'Epoch {i} completed')
        return self.global_best # return best location