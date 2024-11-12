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
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, ann: ANN, num_particles: int) -> None:
        self.ann: ANN = ann
        self.X_train = X_train
        self.y_train = y_train
        self.particles: List[Particle] = []
        self.global_best = -100000000 # initialise global best to very bad
        for i in range(num_particles): self.addParticle(Particle()) ## add our particles
        # set all particles informants
        for particle in self.particles:
            particle.informants = particle.getNewInformants()
        # set fittest informant location (TODO can we check this to just set if informants is not None?)
        for particle in self.particles:
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
            particle_fitness = particle.calculateFitness(x_train, y_train)
            if particle_fitness > self.global_best: pass # TODO :FINSIHED ME
        raise Exception('Not implemented yet')