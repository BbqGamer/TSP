import matplotlib.pyplot as plt
import numpy as np
from numba import njit


class TSP:
    def __init__(self, points, weights):
        self._points = points
        self._weights = weights

        # calculate euclidian distance matrix
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        self.D = np.sqrt(np.sum(diffs**2, axis=-1))
        self.D = np.floor(self.D + 0.5)  # mathematical rounding
        self.D += weights

    @classmethod
    def from_csv(cls, filename: str):
        data = np.genfromtxt(filename, delimiter=";")
        return cls(data[:, :2], data[:, 2])

    def visualize(self, solution=None, title="TSP", outfilename="", labels=False):
        plt.clf()
        x = self._points[:, 0]
        y = self._points[:, 1]
        scatter = plt.scatter(
            x,
            y,
            c=self._weights,
            cmap="YlOrRd",
        )
        plt.colorbar(scatter, label="Cost")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        if solution is not None:
            score = self.score(solution)
            solution = np.concatenate((solution, [solution[0]]))
            plt.plot(
                x[solution],
                y[solution],
                color="black",
                linestyle="-",
                linewidth=1,
                label="Path",
            )  # Highlight the path
            title += f" (score: {score})"

            if labels:
                for i, (xi, yi) in enumerate(self._points):
                    plt.text(
                        xi - 1, yi - 1, str(i), fontsize=6, ha="center", va="center"
                    )

        plt.title(title)

        if outfilename:
            plt.savefig(outfilename, dpi=1000)
        else:
            plt.show()

    def __len__(self) -> int:
        return len(self._points)

    @property
    def solution_size(self) -> int:
        """It is required for us to only use 50% of nodes in solution"""
        return int(np.fix(len(self) / 2))

    def score(self, solution: np.ndarray):
        """Return's the score of the solution"""
        return score(solution, self.D)


@njit()
def score(solution: np.ndarray, D: np.ndarray):
    total_cost = 0
    for i in range(len(solution)):
        total_cost += D[solution[i - 1], solution[i]]
    return total_cost
