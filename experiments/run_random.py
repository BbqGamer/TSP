import argparse
from pathlib import Path

from tsp import TSP
from tsp.solvers import NNWhole, RandomSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve TSP")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    instance = TSP.from_csv(args.input_file)
    solver = NNWhole(instance, args.seed)
    solution = solver.solve()
    instance.visualize(solution)
