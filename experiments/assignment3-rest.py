import csv
from tsp import TSP
import time

from tsp.solvers import (
    solve_greedy_cycle,
    solve_nn_any,
    solve_nn_first,
    solve_regret_greedy_cycle,
    solve_weighted_regret_greedy_cycle,
)

if __name__ == "__main__":
    solvers = [
        solve_greedy_cycle,
        solve_nn_any,
        solve_nn_first,
        solve_regret_greedy_cycle,
        solve_weighted_regret_greedy_cycle,
    ]

    with open("results/assignment3-rest.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            print(f"--- {prob} ---")
            for solver in solvers:
                print(solver.__name__)
                for i in range(200):
                    start = time.perf_counter()
                    res = solver(problem.D, i, problem.solution_size)
                    res_time = time.perf_counter() - start
                    sc = problem.score(res)
                    writer.writerow([prob, solver.__name__, i, sc, res_time, 0])

                