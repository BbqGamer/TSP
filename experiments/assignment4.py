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
from tsp.localsearch.moves import IntraType
import numpy as np
from numba import njit, objmode


StartingMethod = typing.Literal["random"]
LocalSearchMethod = typing.Literal["steepest"]


@njit(cache=False)
def random_start_greedy_experiment(
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
    for i in range(200):
        if starting == "random":
            start_sol, unselected = random_starting_from_starting(n, sol_size, i)
        else:
            start_sol = solve_greedy_cycle(D, i, sol_size)
            unselected = np.array([i for i in range(n) if i not in start_sol])

        with objmode(start="f8"):
            start = time.perf_counter()

        if method == "steepest":
            sol, num_iters = local_search_steepest_candidate_edge(
                start_sol, unselected, D, intra_move
            )
        else:
            sol, num_iters = local_search_greedy(start_sol, unselected, D, intra_move)

        with objmode(end="f8"):
            end = time.perf_counter()

        times.append(end - start)
        scores.append(score(sol, D))
        iters.append(num_iters)
    return scores, times, iters


if __name__ == "__main__":
    # seed numpy for reproducibility
    with open("results/assignment4.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            print(f"--- {prob} ---")
            for intra_move in typing.get_args(IntraType):
                for search_method in typing.get_args(LocalSearchMethod):
                    for start_method in typing.get_args(StartingMethod):
                        method = intra_move + "_" + search_method + "_" + start_method
                        print(method)
                        scores, times, iters = random_start_greedy_experiment(
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
