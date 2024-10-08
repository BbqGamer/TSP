import matplotlib.pyplot as plt
import numpy as np


class TSP:
    def __init__(self, points, weights):
        self._points = points

        # calculate euclidian distance matrix
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        self.D = np.sqrt(np.sum(diffs**2, axis=-1))
        self.D = np.floor(self.D + 0.5)  # mathematical rounding

        self.c = weights

    @classmethod
    def from_csv(cls, filename: str):
        data = np.genfromtxt(filename, delimiter=";")
        return cls(data[:, :2], data[:, 2])

    def visualize(self, solution=None):
        x = self._points[:, 0]
        y = self._points[:, 1]
        scatter = plt.scatter(
            x,
            y,
            c=self.c,
            cmap="YlOrRd",
        )
        plt.colorbar(scatter, label="Cost")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        title = "TSP points to visit with costs"

        if solution is not None:
            solution = np.concatenate((solution, [solution[0]]))
            plt.plot(
                x[solution],
                y[solution],
                color="black",
                linestyle="-",
                linewidth=1,
                label="Path",
            )  # Highlight the path
            title += f" (score: {self.score(solution)})"

        plt.title(title)
        plt.show()

    def __len__(self) -> int:
        return len(self._points)

    def score(self, solution: np.ndarray) -> float:
        """Return's the score of the solution (distance + weights)"""
        # weights
        score = np.sum(self.c[solution])

        # distance
        closed_path = np.concatenate((solution, [solution[0]]))
        index_pairs = np.vstack((closed_path[:-1], closed_path[1:])).T
        score += np.sum(self.D[index_pairs[:, 0], index_pairs[:, 1]])

        return score
