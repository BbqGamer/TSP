import heapq

import numpy as np
from numba import njit

from tsp.localsearch.moves import (
    inter_node_exchange_delta,
    intra_edge_exchange_delta,
)

NULL = np.iinfo(np.uint8).max


@njit()
def local_search_steepest_lazy(sol, unselected, D) -> tuple[np.ndarray, int]:
    num_iterations = 0

    unselected_map = hashmap_array(unselected, len(D))
    solution_matrix = get_edge_matrix(sol, len(D))

    # first iteration - evaluate all moves
    moves_pq = evaluate_all_moves(sol, unselected, D)
    while moves_pq:
        num_iterations += 1
        delta, (move_type, i, a, j, b) = heapq.heappop(moves_pq)
        if move_type == "intra_edge":
            # check if applicable
            pass
            # apply intra-route edge exchange
        else:
            pass
            # apply inter-route node exchange


def evaluate_all_moves(sol, unselected, D):
    """Evaluates all possible improving moves and returns a priority queue of moves"""
    all_moves = []
    n = len(sol)

    # Intra-route edge exchange:
    for i in range(n):
        for j in range(i + 2, n):
            delta = intra_edge_exchange_delta(D, sol, i, j)
            if delta < 0:
                a, b = sol[i], sol[j]
                a_next, b_next = sol[(i + 1) % n], sol[(j + 1) % n]
                all_moves.append((delta, ("intra_edge", (a, a_next), (b, b_next))))

    # Inter-route node exchange:
    for i in range(n):
        for k in range(len(unselected)):
            delta = inter_node_exchange_delta(D, sol, i, unselected, k)
            if delta < 0:
                all_moves.append((delta, ("inter_node", i, sol[i], k, unselected[k])))

    heapq.heapify(all_moves)
    return all_moves


@njit()
def get_edge_matrix(sol, size):
    edges = -np.ones((size, size), dtype=np.uint8)  # table of NULLs
    for i in range(len(sol)):
        edges[sol[i], sol[(i + 1) % len(sol)]] = i
    return edges


@njit()
def hashmap_array(array, size):
    res = np.zeros(size, dtype=np.bool)
    for k in array:
        res[k] = 1
    return res
