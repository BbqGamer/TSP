import numpy as np
from numba import njit


@njit(cache=True)
def random_starting(
    n, sol_size, seed: int | None = 42
) -> tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    points = np.arange(0, n)
    np.random.shuffle(points)
    selected = points[:sol_size]
    unselected = points[sol_size:]
    return selected, unselected


@njit()
def random_starting_from_starting(
    n, sol_size, starting
) -> tuple[np.ndarray, np.ndarray]:
    points = np.arange(0, n)
    np.random.seed(starting)
    np.random.shuffle(points)
    selected = points[:sol_size]
    unselected = points[sol_size:]
    return selected, unselected
