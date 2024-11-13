import numpy as np

from tsp.localsearch.lazy import NULL, array_map, get_edge_matrix


def test_get_edge_matrix():
    sol = [0, 1, 2]
    res = get_edge_matrix(sol, 4)
    assert np.all(
        res
        == np.array(
            [
                [NULL, 0, NULL, NULL],
                [NULL, NULL, 1, NULL],
                [2, NULL, NULL, NULL],
                [NULL, NULL, NULL, NULL],
            ]
        )
    )


def test_array_map():
    unselected = np.array([2, 5, 1])
    res = array_map(unselected, 6)
    assert np.all(res == np.array([NULL, 2, 0, NULL, NULL, 1]))
