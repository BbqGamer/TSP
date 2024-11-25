import csv
import time

import numpy as np

from tsp import TSP
from tsp.largescale import large_scale_neighborhood_search


def start_experiment(
    problem,
    instance_tsp,
):
    np.random.seed(42)

    n_global_iters = 21

    scores = np.zeros(n_global_iters)
    times = np.zeros(n_global_iters)
    iters = np.zeros(n_global_iters)
    best_sol = None
    best_score = np.inf

    if instance_tsp == "TSPA":
        time_limit = 1.927722
    elif instance_tsp == "TSPB":
        time_limit = 2.32237

    for i in range(n_global_iters):
        print("Large scale neighborhood search iteration -", i)

        start = time.perf_counter()

        sol, num_iters = large_scale_neighborhood_search(
            len(problem), problem.solution_size, problem.D, time_limit
        )

        end = time.perf_counter()
        times[i] = end - start

        score = problem.score(sol)
        scores[i] = score
        if score < best_score:
            best_score = score
            best_sol = sol

        iters[i] = num_iters
        print(f"Score: {score}, Time: {times[i] * 1000} ms, Iterations: {num_iters}")
        print(len(sol))

    return scores, times, iters, best_sol


if __name__ == "__main__":
    with open("results/assignment7.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "i", "score", "time", "iter"])
        for prob in ["TSPA", "TSPB"]:
            problem = TSP.from_csv("data/" + prob + ".csv")
            method = "large scale neighborhood search"
            print(f"--- {prob} ---")
            scores, times, iters, best_sol = start_experiment(
                problem,
                prob,
            )

            problem.visualize(
                best_sol,
                title=f"{prob} {method}",
                outfilename=f"results/{prob}_{method}.png",
            )

            print(f" - {sum(times) / len(times) * 1000} ms")
            for i in range(len(scores)):
                writer.writerow([prob, method, i, int(scores[i]), times[i], iters[i]])
