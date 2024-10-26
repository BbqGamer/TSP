from tsp.localsearch.descent import greedy_descent, steepest_descent, IntraType
import numpy as np
from typing import Literal
from numba import njit

LocalSearchMethod = Literal["steepest", "greedy"]


@njit()
def local_search_steepest(sol, unselected, D, intra_move: IntraType) -> np.ndarray:
    while True:
        improved = steepest_descent(sol, unselected, D, intra_move)
        if not improved:
            return sol

@njit()
def local_search_greedy(sol, unselected, D, intra_move: IntraType) -> np.ndarray:
    while True:
        improved = greedy_descent(sol, unselected, D, intra_move)
        if not improved:
            return sol

@njit()
def random_starting(n, sol_size) -> tuple[np.ndarray, np.ndarray]:
    points = np.arange(0, n)
    np.random.shuffle(points)
    selected = points[:sol_size]
    unselected = points[sol_size:]
    return selected, unselected
