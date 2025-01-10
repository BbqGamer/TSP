import time

import numpy as np
from numba import njit
from tqdm import tqdm

from tsp import score
from tsp.solvers import solve_weighted_regret_greedy_cycle
from tsp.utils import random_starting


@njit
def operator_1(x1, x2, D, sol_size):
    """Recombination operator for TSP, takes 2 solutions and returns a single one

    We locate in the offspring all common nodes and edges and fill the rest of
    the solution at random"""
    edges_other = set()
    for i in range(len(x2)):
        edges_other.add((x2[i - 1], x2[i]))

    base_nodes = set()
    base = []
    for i in range(len(x1)):
        if (x1[i - 1], x1[i]) in edges_other:
            left = x1[i - 1]
            right = x1[i]
            if left not in base_nodes:
                base_nodes.add(left)
                base.append(left)
            if right not in base_nodes:
                base_nodes.add(right)
                base.append(right)

    base = base + [-1] * (sol_size - len(base))
    base_set = [0 for i in range(len(D))]
    for i in base:
        base_set[i] = 1

    availible = []
    for i in range(len(D)):
        if base_set[i] == 0:
            availible.append(i)
    availible = np.array(availible)

    np.random.shuffle(availible)
    r = 0
    for i in range(len(base)):
        if base[i] == -1:
            base[i] = availible[r]
            r += 1
    return np.array(base)


@njit
def operator_2(x1, x2, D, sol_size):
    """Recombination operator for TSP, takes 2 solutions and returns a single one

    We choose one of the parents as the starting solution. We remove from this
    solution all edges and nodes that are not present in the other parent.
    The solution is repaired using the weighted regret greedy cycle heuristic.
    We also test the version of the algorithm without local search after recombination
    (we still use local search for the initial population)."""
    edges_other = set()
    for i in range(len(x2)):
        edges_other.add((x2[i - 1], x2[i]))

    base_nodes = set()
    base = []
    for i in range(len(x1)):
        if (x1[i - 1], x1[i]) in edges_other:
            left = x1[i - 1]
            right = x1[i]
            if left not in base_nodes:
                base_nodes.add(left)
                base.append(left)
            if right not in base_nodes:
                base_nodes.add(right)
                base.append(right)

    if not base:
        base = [x1[0]]

    repaired = solve_weighted_regret_greedy_cycle(D, base, sol_size)
    return repaired


def solve_tsp_with_evolutionary(D, solution_size, timeout, popsize=20):
    """Solve TSP with hybrid evolutionary algorithm

    - Elite population of 20
    - Steady state algorithm
    - Parents selected from the population with the uniform propbability
    - There must be no copies of the same solution in the population

    We use 2 recombination operators:
    - operator_1 - intersection and filling rest by random
    - operator_2 - difference of nodes of sol1 and sol2 and repair using heuristic
    """
    # Generate an initial population X
    iters = 0
    X = [
        random_starting(len(D), solution_size, seed=np.random.randint(1000))[0]
        for i in range(popsize)
    ]
    S = [score(x, D) for x in X]

    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        # Draw at random two different solutions (parents) using uniform distribution
        i1, i2 = np.random.choice(popsize, 2, replace=False)
        x1 = X[i1]
        x2 = X[i2]

        # Construct an offspring solution by recombining parents
        # if np.random.random() < 0.5:
        #     y = operator_1(x1, x2, D, solution_size)
        # else:
        y = operator_2(x1, x2, D, solution_size)
        # assert len(set(y)) == solution_size

        # y := Local search (y)
        # unselected = np.array([i for i in range(len(D)) if i not in y])
        # y = local_search_steepest(y, unselected, D, "intra_edge")[0]

        # if y is better than the worst solution in the population and has different score than any other solution
        s = int(score(y, D))
        if s in S:
            continue

        # Add y to the population and remove the worst solution
        worst_i = int(np.argmax(S))
        if s < S[worst_i]:
            X[worst_i] = y
            S[worst_i] = s

        iters += 1
    return X[np.argmin(S)], iters


if __name__ == "__main__":
    import sys

    from tsp import TSP

    for prob in ["TSPA", "TSPB"]:
        problem = TSP.from_csv("data/" + prob + ".csv")
        print("Compilation")
        solve_tsp_with_evolutionary(problem.D, problem.solution_size, 0.1)

        print("Real")
        scores = []
        iters = []
        best_sol = None
        best_sc = np.inf
        for i in tqdm(range(20)):
            sol, it = solve_tsp_with_evolutionary(
                problem.D, problem.solution_size, 2.22
            )
            sc = problem.score(sol)
            if sc < best_sc:
                best_sc = sc
                best_sol = sol
            scores.append(sc)
            iters.append(it)

        problem.visualize(
            best_sol,
            title="EVO SECOND",
            outfilename=f"results/{prob}_{sys.argv[2]}.png",
        )

        print(prob)
        print(sum(scores) / len(scores), min(scores), max(scores))
        print(sum(iters) / len(iters), min(iters), max(iters))
