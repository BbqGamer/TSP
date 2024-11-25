import csv
from tsp import TSP, score
from tsp.localsearch.moves import perturb_sol
from tsp.solvers import solve_greedy_cycle
import typing
from tsp.localsearch import (
    local_search_greedy,
    local_search_steepest,
    local_search_steepest_candidate_edge,
    random_starting,
    random_starting_from_starting,
    u_local_search_steepest,
)
import time
import numpy as np
from numba import njit, objmode

IntraType = typing.Literal["intra_edge"]
StartingMethod = typing.Literal["random"]
LocalSearchMethod = typing.Literal["ILS"]


# @njit(cache=False)
def start_experiment(
    n,
    sol_size,
    D,
    intra_move: IntraType,
    method: LocalSearchMethod,
    instance_tsp,
):
    np.random.seed(42)

    n_global_iters = 21

    scores = np.zeros(n_global_iters)
    times = np.zeros(n_global_iters)
    iters = np.zeros(n_global_iters)

    for i in range(n_global_iters):
        print(f"--- Method {method} iteration {i} ---")
        if method == "MSLS":
            num_iters = 200
            intermediate_times = np.zeros(num_iters)
            intermediate_scores = np.zeros(num_iters)
            for j in range(num_iters):
                starting_hash = i * 1000 + j
                print(f"--- Method {method} start hash {starting_hash} ---")
                start_sol, unselected = random_starting_from_starting(
                    n, sol_size, starting_hash
                )
                with objmode(end="f8"):
                    start = time.perf_counter()
                sol, _ = local_search_steepest(start_sol, unselected, D, intra_move)
                with objmode(end="f8"):
                    end = time.perf_counter()

                intermediate_times[j] = end - start
                intermediate_scores[j] = score(sol, D)
            times[i] = intermediate_times.sum()
            scores[i] = intermediate_scores.max()
            iters[i] = num_iters

        if method == "ILS":
            starting_sol, unselected = random_starting_from_starting(n, sol_size, i)

            if instance_tsp == "TSPA":
                time_limit = 1.927722  # 2 * 200
            elif instance_tsp == "TSPB":
                time_limit = 1.742415  # 1 * 200

            with objmode(end="f8"):
                start = time.perf_counter()

            sol, num_iters = u_local_search_steepest(
                starting_sol, unselected, D, intra_move, start, time_limit
            )
            print(f" - {num_iters} iterations")
            print("SCORE", score(sol, D))

            with objmode(end="f8"):
                end = time.perf_counter()

            times[i] = end - start
            scores[i] = score(sol, D)
            iters[i] = num_iters
    return scores, times, iters


if __name__ == "__main__":
    with open("results/assignment6.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            print(f"--- {prob} ---")
            for intra_move in typing.get_args(IntraType):
                for search_method in typing.get_args(LocalSearchMethod):
                    for start_method in typing.get_args(StartingMethod):
                        method = search_method + "_" + start_method
                        print(method)
                        scores, times, iters = start_experiment(
                            len(problem),
                            problem.solution_size,
                            problem.D,
                            intra_move,
                            search_method,
                            prob,
                        )

                        print(f" - {sum(times) / len(times) * 1000} ms")
                        for i in range(len(scores)):
                            writer.writerow(
                                [prob, method, i, int(scores[i]), times[i], iters[i]]
                            )
