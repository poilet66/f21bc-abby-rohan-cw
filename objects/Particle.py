import numpy as np
from typing import Any, List
from ProblemSpace import ProblemSpace
import random

"""
 NOTE: We still need to figure out how we're going to map from location -> ANN
 A handy function within the ANN class such as tuneFromLocation(location: np.ndarray[]) would be handy
 Essentially either creates (slightly worse performing) or modifies existing ANN as per parameters specified by location
 """
# TODO: Make these hyperparameters
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
        self.informants: List["Particle"] = []
        self.previous_fittest_location = self.location
        self.previous_fittest_informant_location = None
        
        
    def getNewInformants(self) -> List["Particle"]:
        other_particles = [particle for particle in self.problem_space.particles if particle is not self] # get all other particles (can't be its own informant)
        num_informants = random.randint(1, len(other_particles))

        return random.sample(other_particles, num_informants) # get a random sample from other particles

    def updatePosition(self) -> None:
        problem_space = self.problem_space
        # Get new velocity
        self.velocity = self.calculateVelocityChange(problem_space.alpha, problem_space.beta, problem_space.gamma)
        # Add velocity to position
        self.location += self.velocity
    
    def calculateVelocityChange(self, alpha, beta, gamma, delta):
        # Probably gonna want to use problem_space here, need methods within that to find global maximums etc
        newVelocity = np.zeros_like(self.velocity)
        for i in range(len(self.velocity)):
            # generate a random factor for cognitive, social and global components
            b = np.random.uniform(0, beta) #cognitive
            c = np.random.uniform(0, gamma) # social
            d = np.random.uniform(0, delta) # delta

            #calculate components
            inertia = alpha * self.velocity[i]
            cognitive = b * (self.previous_fittest_location[i] - self.location[i])

            if (self.previous_fittest_informant_location != None):
                social = c * (self.previous_fittest_informant_location[i] - self.location[i])
            else:
                social = 0 # no informant influence if no best position

            if (self.problem_space.global_best != None):
                glbl = d * (self.problem_space.global_best[i] - self.location[i])
            else:
                glbl = 0 # no global influence if no global best position

            newVelocity[i] = inertia + cognitive + social + glbl

        return newVelocity
                   



    def fittestInformantLocation(self) -> np.ndarray:
        best_particle = self.informants[0] # assume first informant has best fitness
        for particle in self.informants[1:]:
            # If new particle has better fitness, reassign best particle
            if self.get_fitness(particle.location) > self.get_fitness(best_particle.location):
                best_particle = particle

        return best_particle.location # return best particles location
    
    def update_fields(self) -> None:
        
        # Update personal best location if current is better
        if(self.get_fitness(self.location) > self.get_fitness(self.previous_fittest_location)):
            self.previous_fittest_location = self.location
        # Update previous informant best
        if(self.get_fitness(self.fittestInformantLocation()) > self.get_fitness(self.previous_fittest_informant_location)):
            self.previous_fittest_informant_location = self.fittestInformantLocation() 
        
    def get_fitness(self, location: np.ndarray) -> np.float64:
        return self.problem_space.calculate_fitness(location)
