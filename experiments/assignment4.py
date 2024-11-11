import csv
import time

import numpy as np
from numba import njit, objmode

from tsp import TSP, score
from tsp.localsearch import (
    local_search_steepest,
    random_starting,
)


@njit(cache=True)
def random_start_greedy_experiment(n, sol_size, D, lazy=False):
    np.random.seed(42)

    scores = []
    times = []
    iters = []
    best_sc = float("inf")
    best_sol = None
    for i in range(200):
        start_sol, unselected = random_starting(n, sol_size)

        with objmode(start="f8"):
            start = time.perf_counter()

        if lazy:
            sol, num_iters = local_search_steepest_lazy(start_sol, unselected, D)
        else:
            sol, num_iters = local_search_steepest(
                start_sol, unselected, D, "intra_edge"
            )

        with objmode(end="f8"):
            end = time.perf_counter()

        times.append(end - start)
        sc = score(sol, D)
        scores.append(sc)
        iters.append(num_iters)
        if sc < best_sc:
            best_sc = sc
            best_sol = sol
    return scores, times, iters, best_sol


if __name__ == "__main__":
    with open("results/assignment3.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            print(f"--- {prob} ---")
            for lazy in [True, False]:
                print("Lazy evaluation steepest" if lazy else "Regular steepest")
                scores, times, iters, best_sol = random_start_greedy_experiment(
                    len(problem), problem.solution_size, problem.D, lazy
                )

                title = f"{prob} - Steepest localsearch"
                if lazy:
                    title += " - lazy evaluation"
                plotfile = (
                    f"results/assignment3-plot-{prob}{'-lazy' if lazy else ''}.png"
                )
                problem.visualize(best_sol, title, plotfile)
                print(f" - {sum(times) / len(times) * 1000} ms")
