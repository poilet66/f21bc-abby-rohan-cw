import numpy as np
from typing import Any
from ProblemSpace import ProblemSpace

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
        self.location: np.ndarray[Any, np.dtype[np.float64]] = np.random.uniform(low=INITIAL_LOW_BOUND, high=INTIIAL_HIGH_BOUND, size=problem_space.ann.countParams()) # Init to random location within bounds
        self.velocity: np.ndarray[Any, np.dtype[np.float64]] = np.random.uniform(low=INITIAL_LOW_BOUND, high=INTIIAL_HIGH_BOUND, size=problem_space.ann.countParams()) # Velocity to same, maybe different bounds for this?
        self.currentLoss: np.float64 = np.float64(-1) 
        
    def calculateNextPosition(self):
        raise Exception('Not implemented yet')
    
    def calculateVelocityChange(self):
        # Probably gonna want to use problem_space here, need methods within that to find global maximums etc
        raise Exception('Not implemented yet')

    # Calculate score of current area (loss from using current position as ANN parameters)
    # I.e.: Set ANN weights/biases to current position -> do forward pass -> calculate loss
    def calculateFitness(self):
        self.problem_space.ann.updateParameters(self.location)
        # results = self.problem_space.ann.forward_for() etc.
        # loss = self.problem_space.ann.calculateLoss(results) etc.
        raise Exception('Not implemented yet')