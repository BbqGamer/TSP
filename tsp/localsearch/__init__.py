from tsp.localsearch.descent import steepest_descent
import numpy as np
from numba import njit


@njit()
def local_search_steepest(sol, unselected, D, intra_node: bool) -> np.ndarray:
    while True:
        improved = steepest_descent(sol, unselected, D, intra_node)
        if not improved:
            return sol


@njit()
def random_starting(n, sol_size) -> tuple[np.ndarray, np.ndarray]:
    points = np.arange(0, n)
    np.random.shuffle(points)
    selected = points[:sol_size]
    unselected = points[sol_size:]
    return selected, unselected
