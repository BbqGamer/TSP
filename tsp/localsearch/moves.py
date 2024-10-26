import numpy as np
from typing import Literal


Move = tuple[Literal["intra_node", "intra_edge", "inter_node"], int, int]


def intra_node_exchange(sol, i, j):
    sol[i], sol[j] = sol[j], sol[i]


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
        return D[a_prev, b] + D[b, a] + D[a, b_next] \
            - D[a_prev, a] - D[a, b] - D[b, b_next]

    return D[a_prev, b] + D[b, a_next] + D[b_prev, a] + D[a, b_next] \
        - D[a_prev, a] - D[a, a_next] - D[b_prev, b] - D[b, b_next]


def intra_edge_exchange(sol, i, j):
    sol[i+1:j+1] = np.flip(sol[i+1:j+1])


def intra_edge_exchange_delta(D, sol, i, j):
    """Calculate change in objective function if you exchange i-th and j-th edges from sol"""
    n = len(sol)
    a = sol[i]
    b = sol[j]
    a_next = sol[(i + 1) % n]
    b_next = sol[(j + 1) % n]

    return D[b_next, a_next] + D[a, b] \
        - D[a, a_next] - D[b_next, b]


def inter_node_exchange(sol, i, unselected_nodes, k):
    sol[i], unselected_nodes[k] = unselected_nodes[k], sol[i]


def inter_node_exchange_delta(D, sol, i, unselected_nodes, k):
    """Calculate change in objective function if you exchange nodes sol[i] and some node (not in sol)"""
    a = sol[i]
    a_prev = sol[i - 1]
    a_next = sol[(i + 1) % len(sol)]
    node = unselected_nodes[k]
    return D[a_prev, node] + D[node, a_next] \
        - D[a_prev, a] - D[a, a_next]
