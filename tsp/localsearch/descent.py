import numpy as np

from tsp.localsearch.moves import inter_node_exchange, intra_node_exchange, intra_node_exchange_delta, inter_node_exchange_delta, Move, intra_edge_exchange, intra_edge_exchange_delta
from numba import njit


@njit()
def steepest_descent(sol, unselected, D, intra_node=True) -> bool:
    """Takes one step of greedy descent (using intra and inter node exchange)
    Returns: True if objective function improved and False otherwise"""
    n = len(sol)
    improved = False
    best_delta = 0.0
    best_move: Move | None = None

    # Intra-route node exchange:
    if intra_node:
        for i in range(n):
            for j in range(i + 1, n):
                delta = intra_node_exchange_delta(D, sol, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("intra_node", i, j)
    else:
        for i in range(n):
            for j in range(i + 2, n):
                delta = intra_edge_exchange_delta(D, sol, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("intra_edge", i, j)

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
        elif move_type == "intra_edge":
            intra_edge_exchange(sol, i, j)
        else:
            inter_node_exchange(sol, i, unselected, j)
        improved = True
    return improved
