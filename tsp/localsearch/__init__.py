from typing import Literal

import numpy as np
from numba import njit

from tsp.localsearch.descent import (
    IntraType,
    greedy_descent,
    steepest_descent,
)

LocalSearchMethod = Literal["steepest", "greedy"]


@njit()
def local_search_steepest(
    sol, unselected, D, intra_move: IntraType
) -> tuple[np.ndarray, int, int]:
    num_iterations = 0
    delta_evaluations = 0
    while True:
        improved, delta_evals = steepest_descent(sol, unselected, D, intra_move)
        delta_evaluations += delta_evals
        num_iterations += 1
        if not improved:
            return sol, num_iterations, delta_evaluations


@njit()
def local_search_greedy(
    sol, unselected, D, intra_move: IntraType
) -> tuple[np.ndarray, int]:
    num_iterations = 0
    while True:
        improved = greedy_descent(sol, unselected, D, intra_move)
        num_iterations += 1
        if not improved:
            return sol, num_iterations


@njit()
def random_starting(n, sol_size, seed) -> tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    points = np.arange(0, n)
    np.random.shuffle(points)
    selected = points[:sol_size]
    unselected = points[sol_size:]
    return selected, unselected
