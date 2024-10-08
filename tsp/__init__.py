from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np


class TSP:
    def __init__(self, points, weights):
        self._points = points

        # calculate euclidian distance matrix
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        self.D = np.sqrt(np.sum(diffs**2, axis=-1))

        self.c = weights

    @classmethod
    def from_csv(cls, filename: str):
        data = np.genfromtxt(filename, delimiter=";")
        return cls(data[:, :2], data[:, 2])

    def visualize(self):
        x = self._points[:, 0]
        y = self._points[:, 1]
        scatter = plt.scatter(
            x,
            y,
            c=self.c,
            cmap="YlOrRd",
        )
        plt.colorbar(scatter, label="Cost")
        plt.title("TSP points to visit with costs")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
