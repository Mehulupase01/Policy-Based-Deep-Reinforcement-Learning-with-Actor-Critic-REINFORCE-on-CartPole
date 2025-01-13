import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:
    """
    A class for creating and managing learning curve plots.
    """

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Reward")
        if title:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        """
        Add a curve to the plot.

        Args:
            y (numpy.ndarray): Vector of average reward results.
            label (str, optional): Label for the curve in the plot legend.
        """
        self.ax.plot(y, label=label)

    def set_ylim(self, lower, upper):
        """
        Set the y-axis limits for the plot.

        Args:
            lower (float): Lower limit of the y-axis.
            upper (float): Upper limit of the y-axis.
        """
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        """
        Add a horizontal line to the plot.

        Args:
            height (float): Y-coordinate of the horizontal line.
            label (str): Label for the horizontal line in the plot legend.
        """
        self.ax.axhline(height, ls="--", c="k", label=label)

    def save(self, name="test.png"):
        """
        Save the plot to a file.

        Args:
            name (str, optional): Filename for the saved plot.
        """
        self.ax.legend()
        self.fig.savefig(name, dpi=500)

def smooth(y, window, poly=1):
    """
    Smooth a vector using the Savitzky-Golay filter.

    Args:
        y (numpy.ndarray): Vector to be smoothed.
        window (int): Size of the smoothing window.
        poly (int, optional): Degree of the smoothing polynomial.

    Returns:
        numpy.ndarray: Smoothed vector.
    """
    return savgol_filter(y, window, poly)