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
Consecutive = [6]
LocalSearchMethod = typing.Literal["ILS"]  # , "BasicLS"]  # BasicLS, "ILS"]


# @njit(cache=False)
def start_experiment(
    n,
    sol_size,
    D,
    intra_move: IntraType,
    method: LocalSearchMethod,
    instance_tsp,
    consecutive,
):
    np.random.seed(42)

    n_global_iters = 21

    scores = np.zeros(n_global_iters)
    times = np.zeros(n_global_iters)
    iters = np.zeros(n_global_iters)

    best_sol = None

    for i in range(n_global_iters):
        print(f"--- Method {method} iteration {i} ---")

        if method == "BasicLS":
            start_sol, unselected = random_starting_from_starting(n, sol_size, i)
            with objmode(end="f8"):
                start = time.perf_counter()
            sol, _ = local_search_steepest(start_sol, unselected, D, intra_move)
            with objmode(end="f8"):
                end = time.perf_counter()

            if i == 0 or score(sol, D) < score(best_sol, D):
                best_sol = sol

            times[i] = end - start
            scores[i] = score(sol, D)
            iters[i] = 1

        if method == "MSLS":
            num_iters = 200
            intermediate_times = np.zeros(num_iters)
            intermediate_scores = np.zeros(num_iters)
            intermediate_inner_counter = np.zeros(num_iters)

            for j in range(num_iters):
                starting_hash = i * 1000 + j
                # print(f"--- Method {method} start hash {starting_hash} ---")
                start_sol, unselected = random_starting_from_starting(
                    n, sol_size, starting_hash
                )
                with objmode(end="f8"):
                    start = time.perf_counter()
                sol, one_inner_counter = local_search_steepest(
                    start_sol, unselected, D, intra_move
                )
                with objmode(end="f8"):
                    end = time.perf_counter()

                if j == 0 or score(sol, D) < score(best_sol, D):
                    best_sol = sol

                intermediate_times[j] = end - start
                intermediate_scores[j] = score(sol, D)
                intermediate_inner_counter[j] = one_inner_counter

            times[i] = intermediate_times.sum()
            scores[i] = intermediate_scores.min()
            iters[i] = intermediate_inner_counter.sum()

        if method == "ILS":
            starting_sol, unselected = random_starting_from_starting(n, sol_size, i)

            if instance_tsp == "TSPA":
                time_limit = 0.10  # 1.927722  # 2 * 200
            elif instance_tsp == "TSPB":
                time_limit = 0.10  # 2.32237  # 1 * 200

            with objmode(end="f8"):
                start = time.perf_counter()

            sol, num_iters, inner_counter = u_local_search_steepest(
                starting_sol,
                unselected,
                D,
                intra_move,
                start,
                time_limit,
                consecutive=consecutive,
            )
            if i == 0 or score(sol, D) < score(best_sol, D):
                best_sol = sol
            print(f" - {num_iters} iterations")
            print("SCORE", score(sol, D))

            with objmode(end="f8"):
                end = time.perf_counter()

            times[i] = end - start
            scores[i] = score(sol, D)
            iters[i] = inner_counter

            print(f"BEST SOL {score(best_sol, D)}")
            print(f"{best_sol}")

    return scores, times, iters, best_sol


if __name__ == "__main__":
    with open("results/assignment6.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            print(f"--- {prob} ---")
            for intra_move in typing.get_args(IntraType):
                for search_method in typing.get_args(LocalSearchMethod):
                    for consecutive in Consecutive:
                        method = search_method  # + "_" + str(consecutive)
                        print(method)
                        scores, times, iters, best_sol = start_experiment(
                            len(problem),
                            problem.solution_size,
                            problem.D,
                            intra_move,
                            search_method,
                            prob,
                            consecutive,
                        )
                        problem.visualize(
                            best_sol,
                            title=f"{prob} {method}",
                            outfilename=f"results/{prob}_{method}.png",
                        )

                        print(f" - {sum(times) / len(times) * 1000} ms")
                        for i in range(len(scores)):
                            writer.writerow(
                                [prob, method, i, int(scores[i]), times[i], iters[i]]
                            )
