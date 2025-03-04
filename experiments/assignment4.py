import csv
import time

import numpy as np
from numba import njit, objmode

import tsp
from tsp import TSP, score
from tsp.localsearch import (
    local_search_steepest,
)
from tsp.localsearch.lazy import local_search_steepest_lazy
from tsp.utils import random_starting


@njit()
def random_start_greedy_experiment(n, sol_size, D, lazy=False, seed=None):
    if seed:
        np.random.seed(42)

    scores = []
    times = []
    iters = []
    delta_evals = []
    best_sc = float("inf")
    best_sol = None
    for i in range(200):
        start_sol, unselected = random_starting(n, sol_size, None)

        with objmode(start="f8"):
            start = time.perf_counter()

        if lazy:
            sol, num_iters, evals = local_search_steepest_lazy(start_sol, unselected, D)
        else:
            sol, num_iters, evals = local_search_steepest(
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
        delta_evals.append(evals)
    return scores, times, iters, best_sol, delta_evals


if __name__ == "__main__":
    # To jit compile the functions
    mini = tsp.TSP.from_csv("data/mini.csv")
    start_sol, unselected = random_starting(len(mini), mini.solution_size, 0)
    local_search_steepest_lazy(start_sol, unselected, mini.D)
    start_sol, unselected = random_starting(len(mini), mini.solution_size, 0)
    local_search_steepest(start_sol, unselected, mini.D, "intra_edge")

    with open("results/assignment4.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["problem", "method", "i", "score", "time", "iter", "delta_evals"]
        )
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            print(f"--- {prob} ---")
            for lazy in [True, False]:
                print("Lazy evaluation steepest" if lazy else "Regular steepest")
                scores, times, iters, best_sol, delta_evals = (
                    random_start_greedy_experiment(
                        len(problem), problem.solution_size, problem.D, lazy
                    )
                )

                title = f"{prob} - Steepest localsearch"
                if lazy:
                    title += " - lazy evaluation"
                plotfile = (
                    f"results/assignment3-plot-{prob}{'-lazy' if lazy else ''}.png"
                )
                problem.visualize(best_sol, title, plotfile)
                print(f" - average time:  {sum(times) / len(times) * 1000} ms")
                print(f" - average score: {sum(scores) / len(scores)}")
                print(
                    f" - average number of evaluations: {sum(delta_evals) / len(delta_evals)}"
                )
                for i in range(len(scores)):
                    writer.writerow(
                        [
                            prob,
                            "lazy" if lazy else "steepest",
                            i,
                            scores[i],
                            times[i],
                            iters[i],
                            delta_evals[i],
                        ]
                    )
