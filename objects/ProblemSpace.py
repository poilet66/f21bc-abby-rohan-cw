from Particle import Particle
from main import ANN, Perceptron
from typing import List


class ProblemSpace():
    """
    Encapsulating class for particles + ANN for problem
    May contain methods for calcuating local/global bests, etc.
    """
    def __init__(self, ann: ANN) -> None:
        self.ann: ANN = ann
        self.particles: List[Particle] = []

    def addParticle(self, particle: Particle):
        self.particles.append(particle)
