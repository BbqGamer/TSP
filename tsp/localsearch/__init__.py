from tsp.localsearch.descent import (
    greedy_descent,
    steepest_descent,
    IntraType,
    steepest_descent_candidate_edges,
)
import numpy as np
from typing import Literal
from numba import njit

LocalSearchMethod = Literal["steepest", "greedy"]


@njit()
def local_search_steepest(
    sol, unselected, D, intra_move: IntraType
) -> tuple[np.ndarray, int]:
    num_iterations = 0
    while True:
        improved = steepest_descent(sol, unselected, D, intra_move)
        num_iterations += 1
        if not improved:
            return sol, num_iterations


@njit()
def local_search_steepest_candidate_edge(
    sol, unselected, D, intra_move: IntraType
) -> tuple[np.ndarray, int]:
    num_iterations = 0
    while True:
        # print("steepest_candidate_edge, iteration", num_iterations)
        improved = steepest_descent_candidate_edges(sol, unselected, D, intra_move)
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
def random_starting(n, sol_size) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
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
