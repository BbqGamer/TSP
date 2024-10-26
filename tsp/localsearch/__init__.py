from tsp.localsearch.descent import steepest_descent
import numpy as np


def local_search_steepest(sol, unselected, D) -> np.ndarray:
    while True:
        improved = steepest_descent(sol, unselected, D)
        if not improved:
            return sol


def random_starting(n, sol_size) -> tuple[np.ndarray, np.ndarray]:
    points = np.arange(0, n)
    np.random.shuffle(points)
    selected = points[:sol_size]
    unselected = points[sol_size:]
    return selected, unselected
