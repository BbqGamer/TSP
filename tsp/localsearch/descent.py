import numpy as np

from tsp.localsearch.moves import intra_node_exchange_delta, inter_node_exchange_delta, Move, intra_edge_exchange_delta, apply_move
from numba import njit


@njit()
def steepest_descent(sol, unselected, D, intra_node=True) -> bool:
    """Takes one step of greedy descent (using intra and inter node exchange)
    Returns: True if objective function improved and False otherwise"""
    n = len(sol)
    improved = False
    best_delta = 0.0
    best_move: Move | None = None

    if intra_node:
        # Intra-route node exchange:
        for i in range(n):
            for j in range(i + 1, n):
                delta = intra_node_exchange_delta(D, sol, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("intra_node", i, j)
    else:
        # Intra-route edge exchange:
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
        apply_move(sol, unselected, best_move)
        improved = True
    return improved
