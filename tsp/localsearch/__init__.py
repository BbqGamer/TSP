import time
from typing import Literal

import numpy as np
from numba import njit, objmode

from tsp import score
from tsp.localsearch.descent import (
    IntraType,
    greedy_descent,
    steepest_descent,
    steepest_descent_candidate_edges,
)
from tsp.localsearch.moves import perturb_sol

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


@njit(cache=False)
def u_local_search_steepest(
    sol, unselected, D, intra_move: IntraType, start, time_limit, consecutive
) -> tuple[np.ndarray, int]:
    # Initialization
    num_iterations = 0
    inner_counter = 0
    best_sol = sol.copy()
    best_score = np.inf
    best_sol_unselected = unselected.copy()

    with objmode(end="f8"):
        end = time.perf_counter()

    # Main loop
    while (end - start) < time_limit:
        sol = best_sol.copy()
        unselected = best_sol_unselected.copy()
        num_iterations += 1
        # print(f"--- Iteration {num_iterations} ---")
        # print((time.perf_counter() - start))

        # if num_iterations % 2 == 0:
        # perturb_sol(sol, unselected, "inter_node", consecutive)
        # elif num_iterations % 2 == 1:
        perturb_sol(sol, unselected, "intra_edge", consecutive)

        while True:
            inner_counter += 1
            # print("Improving")
            improved = steepest_descent(sol, unselected, D, intra_move)
            if not improved:
                break

        # score the solution
        current_score = score(sol, D)
        if current_score < best_score:
            best_score = current_score
            best_sol = sol.copy()
            best_sol_unselected = unselected.copy()
        # print(f"Current score: {current_score}")

        with objmode(end="f8"):
            end = time.perf_counter()

    return sol, num_iterations, inner_counter


@njit(cache=False)
def local_search_steepest_candidate_edge(
    sol, unselected, D, closest_nodes
) -> tuple[np.ndarray, int]:
    num_iterations = 200
    while True:
        # print("steepest_candidate_edge, iteration", num_iterations)
        improved = steepest_descent_candidate_edges(sol, unselected, D, closest_nodes)
        num_iterations += 1
        if not improved:
            return sol, num_iterations


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
