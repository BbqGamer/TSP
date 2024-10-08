from tsp import TSP
from tsp.solvers import RandomSolver

if __name__ == "__main__":
    instance = TSP.from_csv("data/TSPA.csv")
    solver = RandomSolver(instance)
    solution = solver.solve()
    instance.visualize(solution)
