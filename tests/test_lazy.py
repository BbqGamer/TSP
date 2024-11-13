import numpy as np

from tsp.localsearch.lazy import NULL, get_edge_matrix, hashmap_array


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


def test_hashmap_array():
    unselected = np.array([2, 5, 1])
    res = hashmap_array(unselected, 6)
    assert np.all(res == np.array([0, 1, 1, 0, 0, 1]))
