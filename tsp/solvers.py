from typing import Protocol

import numpy as np

from tsp import TSP


class Solver(Protocol):
    def __init__(self, problem: TSP): ...
    def solve(self) -> np.ndarray:
        """Solution should be a path of indices, every index should be different
        We assume that there is an edge between all pairs and between last with first
        """
        ...


class RandomSolver(Solver):
    def __init__(self, problem: TSP):
        self.problem = problem

    def solve(self):
        path = np.random.choice(len(self.problem), int(np.fix(len(self.problem) / 2)))
        return path
