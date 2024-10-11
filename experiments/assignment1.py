import csv
import multiprocessing

from tsp import TSP
from tsp.solvers import GreedyCycle, NNHead, NNWhole, RandomSolver


def tsp_task(instance: TSP, filename, solver, starting_node):
    solution = solver.solve()
    score = instance.score(solution)
    row = [
        filename,
        solver.__class__.__name__,
        score,
        starting_node,
        str(solution.tolist()).replace(" ", ""),
    ]
    return row


if __name__ == "__main__":
    args = []
    for filename in ["TSPA.csv", "TSPB.csv"]:
        instance = TSP.from_csv("data/" + filename)
        for solverc in [RandomSolver, NNHead, NNWhole, GreedyCycle]:
            for i in range(200):
                starting_node = i
                if solverc == RandomSolver:
                    solver = solverc(instance)
                    starting_node = -1
                else:
                    solver = solverc(instance, starting_node=i)
                args.append((instance, filename, solver, starting_node))

    with multiprocessing.Pool(16) as p:
        res = p.starmap(tsp_task, args)

    with open("results/assignment1.csv", "w") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["filename","solver","score","starting_node","solution"])
        for row in res:
            writer.writerow(row)
