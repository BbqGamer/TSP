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

        for _ in range(self.problem.solution_size - 1):
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
            self.problem.D.copy(), self.starting_node, self.problem.solution_size
        )

    @staticmethod
    def _solve(D, starting, solution_size):
        current = int(np.argmin(D[starting]))
        solution = [starting, current]

        visited = np.zeros(len(D), dtype=bool)
        visited[current] = True
        visited[solution] = True

        for _ in range(solution_size - 2):
            best_posi = -1
            best_delta = np.inf
            best_nn = -1
            for i, pos in enumerate(solution):
                # find closest neighbor to pos from solution
                unvisited_distances = D[pos][~visited]
                nearest_neighbor_index = np.argmin(unvisited_distances)
                nn = int(np.where(~visited)[0][nearest_neighbor_index])

                nextpos = solution[(i + 1) % len(solution)]
                delta = D[pos][nn] + D[nn][nextpos] - D[pos][nextpos]
                if delta < best_delta:
                    best_posi = i
                    best_delta = delta
                    best_nn = nn
            solution.insert(best_posi + 1, best_nn)
            visited[best_nn] = True

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
        for _ in range(solution_size - 2):
            best_i = -1
            best_j = -1
            best_delta = np.inf
            for i in range(len(solution)):
                # find new candidate
                for j in not_visited:
                    pos = solution[i]
                    nextpos = solution[(i + 1) % len(solution)]
                    delta = D[pos, j] + D[j, nextpos] - D[pos, nextpos]
                    if delta < best_delta:
                        best_i = i
                        best_j = j
                        best_delta = delta

            # Add found best candidate
            solution.insert(best_i + 1, best_j)
            not_visited.remove(best_j)

        return np.array(solution)
