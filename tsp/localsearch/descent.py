import numpy as np

from tsp.localsearch.moves import (
    intra_node_exchange_delta,
    inter_node_exchange_delta,
    Move,
    intra_edge_exchange_delta,
    apply_move,
    IntraType,
)
from numba import njit


@njit()
def steepest_descent(sol, unselected, D, intra_move: IntraType) -> bool:
    """Takes one step of steepest descent using inter-route node exchange
    and for intra-route it uses node exchange or edge exchange
    Returns: True if objective function improved and False otherwise
        function changes sol and unselected"""
    n = len(sol)
    improved = False
    best_delta = 0.0
    best_move: Move | None = None

    if intra_move == "intra_node":
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


@njit()
def greedy_descent(sol, unselected, D, intra_move: IntraType) -> bool:
    """Takes one step of greedy descent using inter-route node exchange
    and for intra-route it uses node exchange or edge exchange
    Returns: True if objective function improved and False otherwise
        function changes sol and unselected
    """
    n = len(sol)
    moves: list[Move] = []
    if intra_move == "intra_node":
        # Intra-route node exchange:
        for i in range(n):
            for j in range(i + 1, n):
                moves.append(("intra_node", i, j))
    else:
        # Intra-route edge exchange:
        for i in range(n):
            for j in range(i + 2, n):
                moves.append(("intra_edge", i, j))

    # Inter-route node exchange:
    for i in range(n):
        for k in range(len(unselected)):
            moves.append(("inter_node", i, k))

    indices = np.arange(len(moves))
    np.random.shuffle(indices)

    improved = False
    for m in indices:
        mov_type, i, j = moves[m]
        if mov_type == "intra_node":
            delta = intra_node_exchange_delta(D, sol, i, j)
        elif mov_type == "intra_edge":
            delta = intra_edge_exchange_delta(D, sol, i, j)
        else:
            delta = inter_node_exchange_delta(D, sol, i, unselected, j)
        if delta < 0.0:
            apply_move(sol, unselected, moves[m])
            improved = True
            break
    return improved
