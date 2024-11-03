from Particle import Particle
from main import ANN, Perceptron
from typing import List
from preprocessing import get_preprocessed_data

x_train, y_train, x_test, y_test = get_preprocessed_data()

class ProblemSpace():
    """
    Encapsulating class for particles + ANN for problem
    May contain methods for calcuating local/global bests, etc.
    """
    def __init__(self, ann: ANN, num_particles: int) -> None:
        self.ann: ANN = ann
        self.particles: List[Particle] = []
        self.global_best = -100000000 # initialise global best to very bad
        for i in range(num_particles): self.addParticle(Particle()) ## add our particles
        # set all particles informants
        for particle in self.particles:
            particle.informants = particle.getNewInformants()
        # set fittest informant location (TODO can we check this to just set if informants is not None?)
        for particle in self.particles:
            particle.previous_fittest_informant_location = particle.fittestInformantLocation()

    def addParticle(self, particle: Particle):
        self.particles.append(particle)

    def doEpoch(self) -> None:
        for particle in self.particles:
            particle_fitness = particle.calculateFitness(x_train, y_train)
            if particle_fitness > self.global_best: pass # TODO :FINSIHED ME
        raise Exception('Not implemented yet')