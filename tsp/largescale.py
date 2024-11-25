import time

import numba
import numpy as np
from numba import objmode

from tsp import score
from tsp.localsearch import local_search_steepest, random_starting
from tsp.solvers import solve_weighted_regret_greedy_cycle


@numba.njit()
def destroy(sol: np.ndarray, removal_fraction: float = 0.3) -> np.ndarray:
    """Remove single segment from solution"""
    n = len(sol)
    num_to_remove = int(n * removal_fraction)
    start = np.random.randint(0, n)
    rolled = np.roll(sol, -start)
    return rolled[num_to_remove:]


@numba.njit()
def large_scale_neighborhood_search(instance_size, sol_size, D, time_limit):
    start_sol, unselected = random_starting(instance_size, sol_size, None)
    sol, _, _ = local_search_steepest(start_sol, unselected, D, "intra_edge")

    with objmode(start="f8"):
        start = time.perf_counter()
    end = start

    num_iters = 0
    while (end - start) < time_limit:
        new_sol = destroy(sol)
        new_sol = solve_weighted_regret_greedy_cycle(D, new_sol, sol_size)
        new_sol, _, _ = local_search_steepest(new_sol, unselected, D, "intra_edge")

        if score(new_sol, D) < score(sol, D):
            sol = new_sol

        with objmode(end="f8"):
            end = time.perf_counter()

        num_iters += 1

    return sol, num_iters
