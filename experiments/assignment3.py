from tsp import TSP, score
import typing
from tsp.localsearch import local_search_greedy, local_search_steepest, random_starting, LocalSearchMethod
from tsp.localsearch.moves import IntraType
import numpy as np
from numba import njit


@njit(cache=True)
def random_start_greedy_experiment(
    n, sol_size, D, intra_move: IntraType, method: LocalSearchMethod
) -> np.ndarray:
    np.random.seed(42)

    best_score = np.inf
    for i in range(200):
        start_sol, unselected = random_starting(n, sol_size)
        if method == "steepest":
            sol = local_search_steepest(start_sol, unselected, D, intra_move)
        else:
            sol = local_search_greedy(start_sol, unselected, D, intra_move)
        cur_score = score(sol, D)
        if cur_score < best_score:
            best_score = cur_score
            best_sol = sol
    return best_sol


if __name__ == "__main__":
    problem = TSP.from_csv("data/TSPA.csv")

    for intra_move in typing.get_args(IntraType):
        for method in typing.get_args(LocalSearchMethod):
            print(intra_move, method)
            best_sol = random_start_greedy_experiment(
                len(problem), problem.solution_size, problem.D, intra_move, method
            )
            problem.visualize(best_sol)