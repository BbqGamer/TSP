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


class NNWhole(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self):
        current = self.starting_node
        solution = [current]
        visited = np.zeros(len(self.problem), dtype=bool)
        visited[self.starting_node] = True

        D = self.problem.D.copy()
        D[:, current] = np.inf
        for _ in range(1, self.problem.solution_size):
            besti = 0
            best_nn = np.argmin(D[solution[besti]])
            best_dist = D[solution[besti], best_nn]
            for i, x in enumerate(solution[1:]):
                nn = np.argmin(D[x])
                dist = D[x][nn]
                if dist < best_dist:
                    besti = i
                    best_nn = nn
                    best_dist = dist

            if besti == 0:
                solution.insert(0, best_nn)
            elif besti == len(solution) - 1:
                solution.append(best_nn)
            else:
                if D[solution[besti - 1], best_nn] < D[solution[besti + 1], best_nn]:
                    solution.insert(besti, best_nn)
                else:
                    solution.insert(besti + 1, best_nn)

            D[:, best_nn] = np.inf
        return np.array(solution)
