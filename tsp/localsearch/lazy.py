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

    U = array_map(unselected, len(D))
    E = get_edge_matrix(sol, len(D))

    # first iteration - evaluate all moves
    moves_pq = evaluate_all_moves(sol, unselected, D)
    while moves_pq:
        num_iterations += 1
        delta, move = heapq.heappop(moves_pq)
        move_type = move[0]
        if move_type == "intra_edge":
            a, a_next, b, b_next = move[1:]
            if E[a, a_next] == NULL or E[b, b_next] == NULL:
                continue  # not applicable, we cannot remove inexistent edges

            # TODO: apply intra-route edge exchange
        else:
            a_prev, a, a_next, node = move[1:]
            if E[a_prev, a] == NULL or E[a, a_next] == NULL or U[node] == NULL:
                continue  # not applicable

            # TODO: apply inter-route node exchange


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
                all_moves.append((delta, ("intra_edge", a, a_next, b, b_next)))

    # Inter-route node exchange:
    for i in range(n):
        for k in range(len(unselected)):
            delta = inter_node_exchange_delta(D, sol, i, unselected, k)
            if delta < 0:
                a = sol[i]
                a_next = sol[(i + 1) % n]
                a_prev = sol[i - 1]
                all_moves.append(
                    (delta, ("inter_node", a_prev, a, a_next, unselected[k]))
                )

    heapq.heapify(all_moves)
    return all_moves


@njit()
def get_edge_matrix(sol, size):
    edges = -np.ones((size, size), dtype=np.uint8)  # table of NULLs
    for i in range(len(sol)):
        edges[sol[i], sol[(i + 1) % len(sol)]] = i
    return edges


@njit()
def array_map(array, size):
    res = -np.ones(size, dtype=np.uint8)
    for i, k in enumerate(array):
        res[k] = i
    return res
