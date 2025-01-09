from tsp import TSP
from tsp.localsearch.moves import inter_node_exchange, inter_node_exchange_delta
from tsp.utils import random_starting

if __name__ == "__main__":
    instance = TSP.from_csv("data/TSPA.csv")
    sol, unselected = random_starting(len(instance), instance.solution_size)

    delta = inter_node_exchange_delta(instance.D, sol, 0, unselected, 0)
    print(delta)
    inter_node_exchange(sol, 1, unselected, 2)
    # inter_node_exchange(sol, len(sol) - 1, unselected, len(unselected) - 1)

    delta = inter_node_exchange_delta(instance.D, sol, 0, unselected, 0)
    print(delta)
