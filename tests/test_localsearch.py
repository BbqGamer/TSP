import numpy as np
import pytest

from tsp import TSP
from tsp.localsearch.moves import inter_node_exchange, inter_node_exchange_delta, intra_edge_exchange, intra_edge_exchange_delta, intra_node_exchange, intra_node_exchange_delta


@pytest.fixture
def instance():
    points = np.array([(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)])
    weights = np.array([-1, 0, 1, 2, 3, 1])
    instance = TSP(points, weights)
    yield instance

@pytest.fixture
def solution():
    return np.array([0, 1, 3, 5, 4, 2])


def test_intra_node(instance, solution):
    score_before = instance.score(solution)  # 14

    i, j = 2, 5
    delta = intra_node_exchange_delta(instance.D, solution, i, j)
    assert delta == intra_node_exchange_delta(instance.D, solution, j, i)

    intra_node_exchange(solution, i, j)  # modify solution
    assert (solution == np.array([0, 1, 2, 5, 4, 3])).all()

    score_after = instance.score(solution)  # 18
    assert delta == score_after - score_before  # 4


def test_intra_node_boundary(instance, solution):
    score_before = instance.score(solution)  # 14

    i, j = 0, 5
    delta = intra_node_exchange_delta(instance.D, solution, i, j)
    assert delta == intra_node_exchange_delta(instance.D, solution, j, i)

    intra_node_exchange(solution, j, i)
    assert (solution == np.array([2, 1, 3, 5, 4, 0])).all()

    score_after = instance.score(solution)
    assert delta == score_after - score_before



def test_intra_edge(instance, solution):
    score_before = instance.score(solution)

    i, j = 0, 3
    delta = intra_edge_exchange_delta(instance.D, solution, i, j)

    intra_edge_exchange(solution, i, j)
    assert (solution == np.array([0, 5, 3, 1, 4, 2])).all()

    score_after = instance.score(solution)
    assert delta == score_after - score_before


def test_intra_edge_touching(instance, solution):
    i, j = 5, 0
    assert intra_edge_exchange_delta(instance.D, solution, i, j) == 0
    intra_edge_exchange(solution, i, j)
    old_solution = solution.copy()
    assert (old_solution == solution).all()  # should not change

def test_intra_edge_touching_2(instance, solution):
    i, j = 0, 5
    assert intra_edge_exchange_delta(instance.D, solution, i, j) == 0
    intra_edge_exchange(solution, i, j)
    assert (np.array([0, 2, 4, 5, 3, 1]) == solution).all()  # just change direction


def test_inter_node(instance):
    solution = np.array([0, 1, 2])
    score_before = instance.score(solution)

    i, node = 2, 3
    delta = inter_node_exchange_delta(instance.D, solution, i, node)

    inter_node_exchange(solution, i, node)
    assert (solution == np.array([0, 1, 3])).all()

    score_after = instance.score(solution)
    assert delta == score_after - score_before


def test_inter_node_beginning(instance):
    solution = np.array([0, 1, 2])
    score_before = instance.score(solution)

    i, node = 0, 3
    delta = inter_node_exchange_delta(instance.D, solution, i, node)

    inter_node_exchange(solution, i, node)
    assert (solution == np.array([3, 1, 2])).all()

    score_after = instance.score(solution)
    assert delta == score_after - score_before


