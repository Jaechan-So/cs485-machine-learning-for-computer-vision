import matplotlib.pyplot as plt


def display():
    plt.show()


class FacePlot:
    def __init__(self, total_plots):
        self._plots_per_row = 4
        self._total_plots = total_plots
        self._current_plots = 0

    def add_plot(self, plot):
        plt.subplot(((self._total_plots - 1) // self._plots_per_row) + 1, 4, ++self._current_plots)
        plt.imshow(plot, cmap='gist_gray')
