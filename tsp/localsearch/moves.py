from typing import Literal

import numpy as np
from numba import njit

Move = tuple[Literal["intra_node", "intra_edge", "inter_node"], int, int]
IntraType = Literal["intra_edge"]


@njit(cache=False)
def apply_intra_move_candidate_edge(sol, i, j, best_prev_or_next):
    if i > j:
        i, j = j, i
    if best_prev_or_next == 0:
        sol[i:j] = np.flip(sol[i:j])
    else:
        sol[i + 1 : j + 1] = np.flip(sol[i + 1 : j + 1])


@njit(cache=False)
def apply_inter_move_candidate_edge(sol, unselected, i, k, best_prev_or_next):
    if best_prev_or_next == 0:
        sol[i - 1], unselected[k] = unselected[k], sol[i - 1]
    else:
        sol[(i + 1) % len(sol)], unselected[k] = unselected[k], sol[(i + 1) % len(sol)]


@njit(cache=False)
def intra_candidate_edge_exchange_delta_prev(D, sol, i, j):
    """Calculate change in objective function if you exchange edges from i-th and j-th nodes"""
    a = sol[i]
    b = sol[j]
    # candidate edge is a,b
    a_prev = sol[i - 1]
    b_prev = sol[j - 1]
    return D[b_prev, a_prev] + D[a, b] - D[a, a_prev] - D[b_prev, b]


@njit(cache=False)
def intra_candidate_edge_exchange_delta_next(D, sol, i, j):
    """Calculate change in objective function if you exchange edges from i-th and j-th nodes"""
    n = len(sol)
    a = sol[i]
    b = sol[j]
    a_next = sol[(i + 1) % n]
    b_next = sol[(j + 1) % n]
    return D[b_next, a_next] + D[a, b] - D[a, a_next] - D[b_next, b]


@njit(cache=False)
def inter_node_candidate_edge_exchange_delta_prev(D, sol, i, unselected_nodes, k):
    """Calculate change in objective function if you exchange nodes sol[i] and some node (not in sol)"""
    a = sol[i]
    a_prev = sol[i - 1]
    a_prev_prev = sol[i - 2]
    node = unselected_nodes[k]
    return D[a_prev_prev, node] + D[node, a] - D[a_prev_prev, a_prev] - D[a_prev, a]


@njit(cache=False)
def inter_node_candidate_edge_exchange_delta_next(D, sol, i, unselected_nodes, k):
    """Calculate change in objective function if you exchange nodes sol[i] and some node (not in sol)"""
    a = sol[i]
    a_next = sol[(i + 1) % len(sol)]
    a_next_next = sol[(i + 2) % len(sol)]
    node = unselected_nodes[k]
    return D[a, node] + D[node, a_next_next] - D[a, a_next] - D[a_next, a_next_next]


@njit(cache=False)
def intra_node_exchange(sol, i, j):
    sol[i], sol[j] = sol[j], sol[i]


@njit(cache=False)
def intra_node_exchange_delta(D, sol, i, j):
    """Calculate change in objective function if you exchange sol[i] and sol[j] nodes"""
    n = len(sol)

    if (j + 1) % n == i:  # needed for adjacent nodes swapping
        i, j = j, i

    a = sol[i]
    b = sol[j]
    a_prev = sol[i - 1]
    b_next = sol[(j + 1) % n]
    a_next = sol[(i + 1) % n]
    b_prev = sol[j - 1]

    if (i + 1) % n == j:
        # Adjacent nodes swapping
        return (
            D[a_prev, b]
            + D[b, a]
            + D[a, b_next]
            - D[a_prev, a]
            - D[a, b]
            - D[b, b_next]
        )

    return (
        D[a_prev, b]
        + D[b, a_next]
        + D[b_prev, a]
        + D[a, b_next]
        - D[a_prev, a]
        - D[a, a_next]
        - D[b_prev, b]
        - D[b, b_next]
    )


@njit(cache=False)
def intra_edge_exchange_2(sol, i, j):
    sol[i + 1 : j + 1] = np.flip(sol[i + 1 : j + 1])


@njit()
def intra_edge_exchange(sol, i, j):
    n = len(sol)
    if j < i:
        j += n

    rolled_indices = np.roll(np.arange(n), -i)

    segment_indices = rolled_indices[1 : j - i + 1]
    sol[segment_indices] = sol[segment_indices][::-1]


@njit(cache=False)
def intra_edge_exchange_delta(D, sol, i, j):
    """Calculate change in objective function if you exchange i-th and j-th edges from sol"""
    n = len(sol)
    a = sol[i]
    b = sol[j]
    a_next = sol[(i + 1) % n]
    b_next = sol[(j + 1) % n]

    return D[b_next, a_next] + D[a, b] - D[a, a_next] - D[b_next, b]


@njit(cache=False)
def inter_node_exchange(sol, i, unselected_nodes, k):
    sol[i], unselected_nodes[k] = unselected_nodes[k], sol[i]


@njit(cache=False)
def inter_node_exchange_delta(D, sol, i, unselected_nodes, k):
    """Calculate change in objective function if you exchange nodes sol[i] and some node (not in sol)"""
    a = sol[i]
    a_prev = sol[i - 1]
    a_next = sol[(i + 1) % len(sol)]
    node = unselected_nodes[k]
    return D[a_prev, node] + D[node, a_next] - D[a_prev, a] - D[a, a_next]


@njit(cache=False)
def apply_move(sol, unselected, best_move):
    move_type, i, j = best_move
    if move_type == "intra_node":
        intra_node_exchange(sol, i, j)
    elif move_type == "intra_edge":
        intra_edge_exchange(sol, i, j)
    else:
        inter_node_exchange(sol, i, unselected, j)


@njit(cache=False)
def perturb_sol(sol, unselected, intra_move: IntraType, num_continous_nodes_affected):
    i = np.random.randint(0, 100)
    for _ in range(num_continous_nodes_affected):
        j = np.random.randint(0, 100)

        # print(f"Perturbing sol, i={i}, j={j}")
        # print(f"i-th node: {sol[i]}, j-th node: {unselected[j]}")
        # print(sol)
        # print(unselected)

        if intra_move == "inter_node":
            inter_node_exchange(sol, i, unselected, j)
        elif intra_move == "intra_edge":
            intra_edge_exchange(sol, i, j)
        i = (i + 1) % 100

        # print(f"Perturbed")
        # print(sol)
        # print(unselected)
