import csv
from tsp import TSP, score
from tsp.solvers import solve_greedy_cycle
import typing
from tsp.localsearch import (
    local_search_greedy,
    local_search_steepest,
    local_search_steepest_candidate_edge,
    random_starting,
    random_starting_from_starting,
)
import time
import numpy as np
from numba import njit, objmode

IntraType = typing.Literal["intra_edge"]
StartingMethod = typing.Literal["random"]
LocalSearchMethod = typing.Literal["MSLS"]


# @njit(cache=False)
def start_experiment(
    n,
    sol_size,
    D,
    intra_move: IntraType,
    method: LocalSearchMethod,
    starting: StartingMethod,
):
    np.random.seed(42)

    scores = []
    times = []
    iters = []

    for i in range(20):
        print(f"--- Method {method} iteration {i} ---")
        if method == "MSLS":
            num_iters = 200
            intermediate_times = []
            intermediate_scores = []
            for j in range(num_iters):
                starting_hash = i * 1000 + j
                print(f"--- Method {method} start hash {starting_hash} ---")
                start_sol, unselected = random_starting_from_starting(
                    n, sol_size, starting_hash
                )
                with objmode(end="f8"):
                    start = time.perf_counter()
                sol, _ = local_search_greedy(start_sol, unselected, D, intra_move)
                with objmode(end="f8"):
                    end = time.perf_counter()

                intermediate_times.append(end - start)
                intermediate_scores.append(score(sol, D))
            times.append(sum(intermediate_times))
            scores.append(max(intermediate_scores))
            iters.append(num_iters)

        # elif method == "ILS":
        #     # Implement ILS here
        #     pass
        #     times.append(end - start)
        #     scores.append(score(sol, D))
        #     iters.append(num_iters)
    return scores, times, iters


if __name__ == "__main__":
    with open("results/assignment6.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPB"]:
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
                            start_method,
                        )

                        print(f" - {sum(times) / len(times) * 1000} ms")
                        for i in range(len(scores)):
                            writer.writerow(
                                [prob, method, i, int(scores[i]), times[i], iters[i]]
                            )
