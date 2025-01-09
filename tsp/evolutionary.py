import time

import numpy as np

from tsp import score
from tsp.localsearch import local_search_steepest
from tsp.utils import random_starting


def operator_1(x1, x2):
    """Recombination operator for TSP, takes 2 solutions and returns a single one

    We locate in the offspring all common nodes and edges and fill the rest of
    the solution at random"""
    return np.array([])


def operator_2(x1, x2):
    """Recombination operator for TSP, takes 2 solutions and returns a single one

    We choose one of the parents as the starting solution. We remove from this
    solution all edges and nodes that are not present in the other parent.
    The solution is repaired using the weighted regret greedy cycle heuristic.
    We also test the version of the algorithm without local search after recombination
    (we still use local search for the initial population)."""
    return np.array([])


def solve_tsp_with_evolutionary(D, solution_size, timeout, popsize=20):
    """Solve TSP with hybrid evolutionary algorithm

    - Elite population of 20
    - Steady state algorithm
    - Parents selected from the population with the uniform propbability
    - There must be no copies of the same solution in the population

    We use 2 recombination operators:
    - operator_1 - intersection and filling rest by random
    - operator_2 - difference of nodes of sol1 and sol2 and repair using heuristic
    """
    # Generate an initial population X
    X = [random_starting(len(D), solution_size)[0] for _ in range(popsize)]
    S = [int(score(x, D)) for x in X]

    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        # Draw at random two different solutions (parents) using uniform distribution
        x1, x2 = X[np.random.choice(popsize, 2, replace=False)]

        # Construct an offspring solution by recombining parents
        if np.random.random() < 0.5:
            y = operator_1(x1, x2)
        else:
            y = operator_2(x1, x2)

        # y := Local search (y)
        unselected = np.array([i for i in range(len(D)) if i not in y])
        y = local_search_steepest(y, unselected, D, "intra_edge")[0]

        # if y is better than the worst solution in the population and has different score than any other solution
        s = int(score(y, D))
        if s in S:
            continue

        # Add y to the population and remove the worst solution
        worst_i = int(np.argmin(S))
        if s < S[worst_i]:
            X[worst_i] = y
            S[worst_i] = s
