import numpy as np

from tsp.localsearch.moves import (
    intra_node_exchange_delta,
    inter_node_exchange_delta,
    Move,
    intra_edge_exchange_delta,
    inter_node_candidate_edge_exchange_delta_prev,
    inter_node_candidate_edge_exchange_delta_next,
    intra_candidate_edge_exchange_delta_prev,
    intra_candidate_edge_exchange_delta_next,
    apply_move,
    apply_intra_move_candidate_edge,
    apply_inter_move_candidate_edge,
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


@njit(cache=False)
def steepest_descent_candidate_edges(sol, unselected, D, intra_move):
    """
    Takes one step of steepest descent using inter-route node exchange
    and for intra-route it uses node exchange or edge exchange.
    Returns: True if objective function improved, False otherwise.
    This function modifies sol and unselected.
    """
    n = len(sol)
    improved = False
    best_delta = 0.0
    best_move_type = -1  # -1: No move, 0: Intra-edge, 1: Inter-node
    best_i = -1
    best_j_or_k = -1
    best_prev_or_next = -1

    # Convert sol to a set-like array for fast membership checking
    sol_mask = np.zeros(len(D), dtype=np.int64)
    for s in sol:
        sol_mask[s] = 1

    # print(sol)

    for i in range(n):
        # print(f"Processing i={i}")
        # Take closest 10 nodes
        closest_nodes = np.argsort(D[sol[i]])[:13]
        # print(f"Closest nodes: {closest_nodes}")

        # Separate intra and inter nodes based on sol_mask
        intra_closest_nodes = [
            node
            for node in closest_nodes
            if sol_mask[node] == 1
            and node != sol[i]
            and node != sol[(i + 1) % n]
            and node != sol[(i - 1) % n]
        ]
        inter_closest_nodes = [node for node in closest_nodes if sol_mask[node] == 0]

        # print(f"Intra closest nodes: {intra_closest_nodes}")
        # print(f"Inter closest nodes: {inter_closest_nodes}")

        # Intra-route edge exchange
        for j_node in intra_closest_nodes:
            j = np.where(sol == j_node)[0][0]
            # print(f"Processing j={j}, j_node={j_node}")
            for prev_or_next in [0, 1]:
                if prev_or_next == 0:
                    delta = intra_candidate_edge_exchange_delta_prev(D, sol, i, j)
                else:
                    delta = intra_candidate_edge_exchange_delta_next(D, sol, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_move_type = 0
                    best_i = i
                    best_j_or_k = j
                    best_prev_or_next = prev_or_next

        # Inter-route edge exchange
        for k_node in inter_closest_nodes:
            k = np.where(unselected == k_node)[0][0]
            # print(f"Processing k={k}, k_node={k_node}")
            for prev_or_next in [0, 1]:
                if prev_or_next == 0:
                    delta = inter_node_candidate_edge_exchange_delta_prev(
                        D, sol, i, unselected, k
                    )
                else:
                    delta = inter_node_candidate_edge_exchange_delta_next(
                        D, sol, i, unselected, k
                    )
                if delta < best_delta:
                    best_delta = delta
                    best_move_type = 1
                    best_i = i
                    best_j_or_k = k
                    best_prev_or_next = prev_or_next
        # raise NotImplementedError("Implement this function")

    if best_move_type != -1:
        if best_move_type == 0:
            # Apply intra-edge move
            # print(
            #     "Applying intra-edge move, best_i, best_j_or_k, best_prev_or_next",
            #     best_i,
            #     best_j_or_k,
            #     best_prev_or_next,
            # )
            apply_intra_move_candidate_edge(sol, best_i, best_j_or_k, best_prev_or_next)
        elif best_move_type == 1:
            # Apply inter-node move
            # print(
            #     "Applying inter-node move, best_i, best_j_or_k, best_prev_or_next",
            #     best_i,
            #     best_j_or_k,
            #     best_prev_or_next,
            # )
            apply_inter_move_candidate_edge(
                sol, unselected, best_i, best_j_or_k, best_prev_or_next
            )
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
