from Particle import Particle
from main import ANN, Perceptron
from typing import List

class ProblemSpace():
    def __init__(self, ann: ANN) -> None:
        self.ann: ANN = ann
        self.particles: List[Particle] = []

    def addParticle(self, particle: Particle):
        self.particles.append(particle)

    # Calculate how 'dimensions' our problemspace is
    # This is basically the number of all weights + biases in the network
    # TODO: Define this
    def getDimensions(self):
        for layer in self.ann.layers:
            pass