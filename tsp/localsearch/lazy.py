import heapq
import sys

import numpy as np
from numba import njit

from tsp.localsearch import random_starting
from tsp.localsearch.moves import (
    inter_node_exchange_delta,
    intra_edge_exchange_delta,
)

NULL = np.iinfo(np.int16).max

NODE = 0
EDGE = 1


@njit()
def add_edge_exchanges_for_edge(heap, D, sol, i):
    n = len(sol)
    for j in range(n):
        if abs(i - j) < 2:
            continue

        a = sol[i]
        b = sol[j]
        a_next = sol[(i + 1) % n]
        b_next = sol[(j + 1) % n]

        delta = intra_edge_exchange_delta(D, sol, i, j)
        if delta < 0:
            heapq.heappush(heap, (delta, (EDGE, a, a_next, b, b_next)))

        # Reversed direction
        delta = intra_edge_exchange_delta(D, sol, j, i)
        if delta < 0:
            heapq.heappush(heap, (delta, (EDGE, b, b_next, a, a_next)))


@njit()
def add_node_exchanges_for_node_from_sol(heap, D, sol, unselected, i):
    for k in range(len(unselected)):
        delta = inter_node_exchange_delta(D, sol, i, unselected, k)
        if delta < 0:
            a = sol[i]
            n = len(sol)
            a_next = sol[(i + 1) % n]
            a_prev = sol[i - 1]
            heapq.heappush(heap, (delta, (NODE, a_prev, a, a_next, unselected[k])))


@njit()
def add_node_exchanges_for_node_from_unselected(heap, D, sol, unselected, k):
    for i in range(len(sol)):
        delta = inter_node_exchange_delta(D, sol, i, unselected, k)
        if delta < 0:
            a = sol[i]
            n = len(sol)
            a_next = sol[(i + 1) % n]
            a_prev = sol[i - 1]
            heapq.heappush(heap, (delta, (NODE, a_prev, a, a_next, unselected[k])))


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
        if move_type == EDGE:
            a, a_next, b, b_next = move[1:]
            if E[a, a_next] == NULL or E[b, b_next] == NULL:
                continue  # not applicable, we cannot remove inexistent edges

            i, j = E[a, a_next], E[b, b_next]

            # update edge matrix
            E[a, a_next] = NULL
            E[a, b] = i
            E[b, b_next] = NULL
            E[a_next, b_next] = j

            # Reverse the edges between the two nodes
            xj, xi = j, i
            n = len(sol)
            if xj < xi:
                xj += n
            rolled_arr = np.roll(sol, -int(xi))

            for x in range(1, xj - xi):
                left, right = rolled_arr[x], rolled_arr[x + 1]
                E[left, right] = NULL
                E[right, left] = (j - x) % n

            rolled_arr[1 : xj - xi + 1] = rolled_arr[1 : xj - xi + 1][::-1]
            sol = np.roll(rolled_arr, int(xi))

            # Add new moves to the priority queue
            for x in range(i, j + 1):
                add_edge_exchanges_for_edge(moves_pq, D, sol, x)
                add_node_exchanges_for_node_from_sol(moves_pq, D, sol, unselected, x)
            add_node_exchanges_for_node_from_sol(
                moves_pq, D, sol, unselected, (j + 1) % n
            )

        else:
            a_prev, a, a_next, node = move[1:]
            if E[a_prev, a] == NULL or E[a, a_next] == NULL or U[node] == NULL:
                continue  # not applicable

            i = E[a, a_next]
            k = U[node]
            sol[i], unselected[k] = node, a

            # update edge matrix and unselected map
            E[a_prev, a] = NULL
            E[a, a_next] = NULL
            E[a_prev, node] = (i - 1) % len(sol)
            E[node, a_next] = i
            U[node] = NULL
            U[a] = k

            add_edge_exchanges_for_edge(moves_pq, D, sol, (i - 1) % len(sol))
            add_edge_exchanges_for_edge(moves_pq, D, sol, i)
            add_node_exchanges_for_node_from_sol(moves_pq, D, sol, unselected, i)
            add_node_exchanges_for_node_from_unselected(moves_pq, D, sol, unselected, k)

        # Sanity check
        for i, node in enumerate(sol):
            assert U[node] == NULL
            assert E[node, sol[(i + 1) % len(sol)]] == i

        for i, node in enumerate(unselected):
            assert U[node] == i

    return sol, num_iterations


@njit()
def evaluate_all_moves(sol, unselected, D):
    """Evaluates all possible improving moves and returns a priority queue of moves"""
    all_moves = []
    n = len(sol)

    # Intra-route edge exchange:
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 2:
                continue
            delta = intra_edge_exchange_delta(D, sol, i, j)
            if delta >= 0:
                continue

            a, b = sol[i], sol[j]
            a_next, b_next = sol[(i + 1) % n], sol[(j + 1) % n]
            all_moves.append((delta, (EDGE, a, a_next, b, b_next)))

    # Inter-route node exchange:
    for i in range(n):
        for k in range(len(unselected)):
            delta = inter_node_exchange_delta(D, sol, i, unselected, k)
            if delta < 0:
                a = sol[i]
                a_next = sol[(i + 1) % n]
                a_prev = sol[i - 1]
                all_moves.append((delta, (NODE, a_prev, a, a_next, unselected[k])))

    heapq.heapify(all_moves)
    return all_moves


@njit()
def get_edge_matrix(sol, size):
    edges = np.ones((size, size), dtype=np.int16) * NULL
    for i in range(len(sol)):
        edges[sol[i], sol[(i + 1) % len(sol)]] = i
    return edges


@njit()
def array_map(array, size):
    res = np.ones(size, dtype=np.int16) * NULL
    for i, k in enumerate(array):
        res[k] = i
    return res


if __name__ == "__main__":
    from tsp import TSP

    instance = TSP.from_csv("data/TSPA.csv")
    sol, unselected = random_starting(len(instance), len(instance) - 1, seed=42)
    if len(sys.argv) == 2 and sys.argv[1] == "lazy":
        sol, num_iterations = local_search_steepest_lazy(sol, unselected, instance.D)
        instance.visualize(sol)
    else:
        from tsp.localsearch import local_search_steepest

        asol, num_iterations = local_search_steepest(
            sol.copy(), unselected.copy(), instance.D, "intra_edge"
        )
        instance.visualize(asol)
