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
    def __init__(self, problem: TSP, seed=None):
        self.problem = problem
        self.seed = seed

    def solve(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        path = np.random.choice(len(self.problem), self.problem.solution_size)
        return path


class NNHead(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self):
        current = self.starting_node
        solution = [current]
        visited = np.zeros(len(self.problem), dtype=bool)
        visited[self.starting_node] = True

        for _ in range(1, self.problem.solution_size):
            unvisited_distances = self.problem.D[current][~visited]
            nearest_neighbor_index = np.argmin(unvisited_distances)
            nearest_neighbor = np.where(~visited)[0][nearest_neighbor_index]

            solution.append(nearest_neighbor)
            visited[nearest_neighbor] = True
            current = nearest_neighbor

        return np.array(solution)
