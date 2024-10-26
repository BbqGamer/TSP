import numpy as np

from tsp.localsearch.moves import inter_node_exchange, intra_node_exchange, intra_node_exchange_delta, inter_node_exchange_delta, Move
from numba import njit


@njit()
def steepest_descent(sol, unselected, D) -> bool:
    """Takes one step of greedy descent (using intra and inter node exchange)
    Returns: True if objective function improved and False otherwise"""
    n = len(sol)
    improved = False
    best_delta = 0.0
    best_move: Move | None = None

    # Intra-route node exchange:
    for i in range(n):
        for j in range(i + 1, n):
            delta = intra_node_exchange_delta(D, sol, i, j)
            if delta < best_delta:
                best_delta = delta
                best_move = ("intra_node", i, j)

    # Inter-route node exchange:
    for i in range(n):
        for k in range(len(unselected)):
            delta = inter_node_exchange_delta(D, sol, i, unselected, k)
            if delta < best_delta:
                best_delta = delta
                best_move = ("inter_node", i, k)

    if best_move is not None:
        move_type, i, j = best_move
        if move_type == "intra_node":
            intra_node_exchange(sol, i, j)
        elif move_type == "inter_node":
            inter_node_exchange(sol, i, unselected, j)
        improved = True
    return improved
