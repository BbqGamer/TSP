from tsp import TSP, score
import typing
from tsp.localsearch import local_search_greedy, local_search_steepest, random_starting, LocalSearchMethod
import time
from tsp.localsearch.moves import IntraType
import numpy as np
from numba import njit, objmode


@njit(cache=True)
def random_start_greedy_experiment(
    n, sol_size, D, intra_move: IntraType, method: LocalSearchMethod
):
    np.random.seed(42)

    scores = []
    times = []
    iters = []
    for i in range(200):
        start_sol, unselected = random_starting(n, sol_size)
        with objmode(start='f8'):
            start = time.perf_counter()

        if method == "steepest":
            sol, num_iters = local_search_steepest(start_sol, unselected, D, intra_move)
        else:
            sol, num_iters = local_search_greedy(start_sol, unselected, D, intra_move)

        with objmode(end='f8'):
            end = time.perf_counter()

        times.append(end - start)
        scores.append(score(sol, D))
        iters.append(num_iters)
    return scores, times, iters


if __name__ == "__main__":
    for file in ["data/TSPA.csv", "data/TSPB.csv"]:
        problem = TSP.from_csv(file)
        print(f"--- {file} ---")
        for intra_move in typing.get_args(IntraType):
            for method in typing.get_args(LocalSearchMethod):
                print(" ", intra_move, method)
                scores, times, iters = random_start_greedy_experiment(
                    len(problem), problem.solution_size, problem.D, intra_move, method
                )
                print(f"   score:\t{int(min(scores))}\t{int(sum(scores)/len(scores))}\t{int(max(scores))}")
                print(f"   times:\t{min(times):.3f}\t{sum(times)/len(times):.3f}\t{max(times):.3f}")
                print(f"   iters:\t{int(min(iters))}\t{int(sum(iters)/len(iters))}\t{int(max(iters))}")
