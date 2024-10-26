from tsp import TSP, score
from tsp.localsearch import local_search_steepest, random_starting
import numpy as np
from numba import njit


@njit(cache=True)
def random_start_greedy_experiment(n, sol_size, D) -> np.ndarray:
    np.random.seed(42)

    best_score = np.inf
    for i in range(200):
        start_sol, unselected = random_starting(n, sol_size)
        sol = local_search_steepest(start_sol, unselected, D)
        cur_score = score(sol, D)
        if cur_score < best_score:
            best_score = cur_score
            best_sol = sol
    return best_sol


if __name__ == "__main__":
    problem = TSP.from_csv("data/TSPA.csv")

    best_sol = random_start_greedy_experiment(
        len(problem), problem.solution_size, problem.D)

    problem.visualize(best_sol)
