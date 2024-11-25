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
        return random_solve(self.problem.D, 0, self.problem.solution_size)


@njit()
def random_solve(D, starting, solution_size):
    return np.random.choice(len(D), solution_size, replace=False)


@njit()
def pairwise_circular(lst):
    for i in range(len(lst) - 1):
        yield lst[i], lst[i + 1]
    yield lst[-1], lst[0]


class NNHead(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self):
        return solve_nn_first(
            self.problem.D, self.starting_node, self.problem.solution_size
        )


@njit()
def solve_nn_first(D, starting, sol_size):
    D = D.copy()
    current = starting
    S = [current]
    D[:, current] = np.inf
    for _ in range(sol_size - 1):
        nn = np.argmin(D[current])
        S.append(nn)
        current = nn
        D[:, current] = np.inf
    return np.array(S)


class NNWhole(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self):
        return solve_nn_any(
            self.problem.D.copy(), self.starting_node, self.problem.solution_size
        )


@njit()
def solve_nn_any(D, starting, solution_size):
    DC = D.copy()
    solution = [starting]
    DC[:, starting] = np.inf

    for _ in range(solution_size - 1):
        best_i, best_nn, best_delta = -1, -1, np.inf
        for i, (first, second) in enumerate(pairwise_circular(solution)):
            nn = np.argmin(DC[first])
            delta = D[first][nn] + D[nn][second] - D[first][second]
            if delta < best_delta:
                best_i, best_delta, best_nn = i, delta, nn
        solution.insert(best_i + 1, best_nn)
        DC[:, best_nn] = np.inf
    return np.array(solution)


class GreedyCycle(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self) -> np.ndarray:
        return solve_greedy_cycle(
            self.problem.D,
            self.starting_node,
            self.problem.solution_size,
        )


@njit()
def solve_greedy_cycle(D, starting, solution_size):
    solution = [starting]
    visited = np.zeros(len(D))
    visited[starting] = 1

    for _ in range(solution_size - 1):
        best_i, best_node, best_delta = -1, -1, np.inf
        for i, (first, second) in enumerate(pairwise_circular(solution)):
            for node in np.where(visited == 0)[0]:
                delta = D[first, node] + D[node, second] - D[first, second]
                if delta < best_delta:
                    best_i, best_node, best_delta = i, node, delta
        solution.insert(best_i + 1, best_node)
        visited[best_node] = 1
    return np.array(solution)


class RegretGreedyCycle(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self) -> np.ndarray:
        return solve_regret_greedy_cycle(
            self.problem.D,
            self.starting_node,
            self.problem.solution_size,
        )


@njit()
def solve_regret_greedy_cycle(D, starting, solution_size):
    solution = [starting]
    visited = np.zeros(len(D))
    visited[starting] = 1

    for _ in range(solution_size - 1):
        best_i, best_node, best_score = -1, -1, np.inf
        for node in np.where(visited == 0)[0]:
            bests = []
            for i, (first, second) in enumerate(pairwise_circular(solution)):
                delta = D[first, node] + D[node, second] - D[first, second]
                bests.append((delta, i))
            if len(bests) == 1:
                best_node = node
            else:
                bests = sorted(bests, key=lambda x: x[0])
                regret = bests[0][0] - bests[1][0]
                if regret < best_score:
                    best_i, best_node, best_score = bests[0][1], node, regret
        solution.insert(best_i + 1, best_node)
        visited[best_node] = 1
    return np.array(solution)


class WeightedRegretGreedyCycle(Solver):
    def __init__(self, problem: TSP, starting_node: int):
        self.problem = problem
        self.starting_node = starting_node

    def solve(self) -> np.ndarray:
        return solve_weighted_regret_greedy_cycle(
            self.problem.D,
            self.starting_node,
            self.problem.solution_size,
        )


@njit()
def solve_weighted_regret_greedy_cycle(D, starting: int | np.ndarray, solution_size):
    visited = np.zeros(len(D))

    if isinstance(starting, np.ndarray):
        solution = starting.tolist()
        for node in starting:
            visited[node] = 1
    else:
        solution = [starting]
        visited[starting] = 1

    toadd = solution_size - len(solution)

    for _ in range(toadd):
        best_i, best_node, best_score = -1, -1, np.inf
        for node in np.where(visited == 0)[0]:
            bests = []
            for i, (first, second) in enumerate(pairwise_circular(solution)):
                delta = D[first, node] + D[node, second] - D[first, second]
                bests.append((delta, i))
            if len(bests) == 1:
                best_node = node
            else:
                bests = sorted(bests, key=lambda x: x[0])
                regret = bests[0][0] - bests[1][0]
                curr_delta = bests[0][0]
                score = 0.5 * regret + 0.5 * curr_delta
                if score < best_score:
                    best_i, best_node, best_score = bests[0][1], node, score
        solution.insert(best_i + 1, best_node)
        visited[best_node] = 1
    return np.array(solution)
