def operator_1(sol1, sol2):
    """Recombination operator for TSP, takes 2 solutions and returns a single one

    We locate in the offspring all common nodes and edges and fill the rest of
    the solution at random"""
    ...


def operator_2(sol1, sol2):
    """Recombination operator for TSP, takes 2 solutions and returns a single one

    We choose one of the parents as the starting solution. We remove from this
    solution all edges and nodes that are not present in the other parent.
    The solution is repaired using the weighted regret greedy cycle heuristic.
    We also test the version of the algorithm without local search after recombination
    (we still use local search for the initial population).
    """
    ...


def solve_tsp_with_evolutionary(D, starting, solution_size):
    """Solve TSP with hybrid evolutionary algorithm

    - Elite population of 20
    - Steady state algorithm
    - Parents selected from the population with the uniform propbability
    - There must be no copies of the same solution in the population

    We use 2 recombination operators:
    - operator_1 - intersection and filling rest by random
    - operator_2 - difference of nodes of sol1 and sol2 and repair using heuristic
    """
    ...
