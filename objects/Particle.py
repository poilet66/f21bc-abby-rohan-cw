import numpy as np
from typing import Any, List
from ProblemSpace import ProblemSpace
import random

"""
 NOTE: We still need to figure out how we're going to map from location -> ANN
 A handy function within the ANN class such as tuneFromLocation(location: np.ndarray[]) would be handy
 Essentially either creates (slightly worse performing) or modifies existing ANN as per parameters specified by location
 """
INITIAL_LOW_BOUND = np.float64(-1)
INTIIAL_HIGH_BOUND = np.float64(1)

class Particle():
    """
    Particle within problemspace, do these need to store their fitness? Maybe.
    """
    def __init__(self, problem_space: ProblemSpace) -> None:
        self.problem_space: ProblemSpace = problem_space # Reference to problem space (so we can access local/global maximums etc)
        self.location: np.ndarray[Any, np.dtype[np.float64]] = np.random.uniform(
            low=INITIAL_LOW_BOUND, high=INTIIAL_HIGH_BOUND, size=problem_space.ann.countParams()
        ) # Init to random location within bounds
        self.velocity: np.ndarray[Any, np.dtype[np.float64]] = np.random.uniform(
            low=INITIAL_LOW_BOUND, high=INTIIAL_HIGH_BOUND, size=problem_space.ann.countParams()
        ) # Velocity to same, maybe different bounds for this?
        self.currentLoss: np.float64 = np.float64(-1)
        self.informants: List["Particle"] = []
        self.previous_fittest_location = self.location
        self.previous_fittest_informant_location = None
        
        
    def getNewInformants(self) -> List["Particle"]:
        other_particles = [particle for particle in self.problem_space.particles if particle is not self] # get all other particles (can't be its own informant)
        num_informants = random.randint(1, len(other_particles))

        return random.sample(other_particles, num_informants) # get a random sample from other particles

    def updatePosition(self) -> None:
        self.location += self.velocity
    
    def calculateVelocityChange(self):
        # Probably gonna want to use problem_space here, need methods within that to find global maximums etc
        raise Exception('Not implemented yet')

    # Calculate score of current area (loss from using current position as ANN parameters)
    # I.e.: Set ANN weights/biases to current position -> do forward pass -> calculate loss
    def calculateFitness(self, y_train, y_pred):
        #update parameters based on particle position
        self.problem_space.ann.updateParameters(self.location)

        #calculate the loss using ANN claculate loss method
        self.currentLoss = self.problem_space.ann.calculate_loss(y_train, y_pred)
        self.currentFitness = 1 / (1 + self.currentLoss)  # ensures fitness is between 0 and 1 (1 is best)
        return self.currentFitness
    
    def fittestInformantLocation(self) -> np.ndarray:
        best_particle = self.informants[0] # assume first informant has best fitness
        for particle in self.informants[1:]:
            best_particle = max(best_particle.calculateFitness(), particle.calculateFitness()) # if higher, reassign
        return best_particle.location # return best particles location