from typing import Protocol

import numpy as np
from numba import njit

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
        path = np.random.choice(
            len(self.problem), self.problem.solution_size, replace=False
        )
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


class NNWhole(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self):
        return NNWhole._solve(
            self.problem.D, self.starting_node, self.problem.solution_size
        )

    @staticmethod
    @njit()
    def _solve(D, starting, solution_size):
        current = starting
        solution = [current]

        # Find nearest neighbor considering all positions from current solution
        D[:, current] = np.inf  # prevent finding it as best new node
        for _ in range(1, solution_size):
            best_posi = -1
            best_dist = np.inf
            best_nn = -1
            for i, pos in enumerate(solution):
                nn = np.argmin(D[pos])
                dist = D[pos][nn]
                if dist < best_dist:
                    best_posi = i
                    best_dist = dist
                    best_nn = int(nn)
            solution.insert(best_posi, best_nn)
            D[:, best_nn] = np.inf

        return np.array(solution)


class GreedyCycle(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self) -> np.ndarray:
        return GreedyCycle._solve(
            self.problem.D,
            self.starting_node,
            len(self.problem),
            self.problem.solution_size,
        )

    @staticmethod
    @njit()
    def _solve(D, starting, problem_size, solution_size):
        current = int(np.argmin(D[starting]))
        solution = [starting, current]
        not_visited = set(range(problem_size))
        not_visited.remove(starting)
        not_visited.remove(current)

        # Find nearest neighbor considering all positions from current solution
        for _ in range(1, solution_size):
            best_i = -1
            best_j = -1
            best_delta = np.inf
            for i in range(len(solution) - 1):
                # find new candidate
                for j in not_visited:
                    delta = (
                        D[solution[i], j]
                        + D[j, solution[i + 1]]
                        - D[solution[i], solution[i + 1]]
                    )
                    if delta < best_delta:
                        best_i = i
                        best_j = j
                        best_delta = delta

            # Add found best candidate
            solution.insert(best_i + 1, best_j)
            not_visited.remove(best_j)

        return np.array(solution)
