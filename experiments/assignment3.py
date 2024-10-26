from tsp import TSP
from tsp.localsearch import local_search_steepest, random_starting
import numpy as np

if __name__ == "__main__":
    problem = TSP.from_csv("data/TSPA.csv")

    sol, unselected = random_starting(len(problem), problem.solution_size)
    problem.visualize(sol)

    solution = local_search_steepest(sol, unselected, problem.D)

    problem.visualize(solution)
