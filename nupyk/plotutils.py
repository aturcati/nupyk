# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from .sedreader import SEDReader

sns.color_palette("colorblind")


class BasePlot(object, metaclass=ABCMeta):
    """
        Base Class to plot prediction vs. target
    """

    def __init__(self, ax):
        self._ax = ax

    @abstractmethod
    def plot(self):
        pass

    @property
    def ax(self):
        """
            Getter function for the ax
        """
        return self._ax


class BaseSourcePlot(BasePlot):
    """
        Base Class to plot source data
    """

    def __init__(self, ax, source_obj):
        super().__init__(ax)
        self._source_obj = source_obj
        self.plot()

    def plot(self):
        pass

    @property
    def source_obj(self):
        """
            Getter function for the ax
        """
        return self._source_obj


class SEDReaderSourcePlot(BaseSourcePlot):
    """
        Plot a SEDReader Object
    """

    def __init__(self, ax, source_obj: SEDReader):
        super().__init__(ax, source_obj)

    def plot(self):
        df = self.source_obj.dataframe
        self._ax.errorbar(
            df["freq"],
            df["flux"],
            yerr=[df["flux_plus"] - df["flux"], df["flux"] - df["flux_min"]],
            ls="",
            marker=".",
            alpha=0.2,
            label="Raw Data",
        )

        self._ax.set_xlim([1e8, 1e28])
        self._ax.set_ylim([1e-15, 1e-8])
        self._ax.set_xlabel("Frequency [Hz]")
        self._ax.set_ylabel("nuFnu [erg/cm2/s]")
        self._ax.loglog()


class ProcessedSourcePlot(BaseSourcePlot):
    """
        Plot a row of a processed_dataframe
    """

    def __init__(self, ax, source_obj: SEDReader, freq_bins_names: list):
        self._frequency_bins_names = freq_bins_names
        super().__init__(ax, source_obj)

    def plot(self):
        x_plot = []
        y_plot = []
        for n in self._frequency_bins_names:
            x_plot.append(f"{n}_freq_mean")
            y_plot.append(f"{n}_flux_mean")

        source = self._source_obj

        for i, cp in enumerate(zip(x_plot, y_plot)):
            xn, yn = cp
            self._ax.plot(
                source[xn],
                source[yn],
                marker="o",
                markersize=10,
                label="Proc. Data ({0})".format(xn.split("_")[0]),
            )

        self._ax.set_xlim([1e8, 1e28])
        self._ax.set_ylim([1e-15, 1e-8])
        self._ax.set_title(source["name"])
        self._ax.set_xlabel("Frequency [Hz]")
        self._ax.set_ylabel("nuFnu [erg/cm2/s]")
        self._ax.loglog()


class PredVsTargetPlot(BasePlot):
    """
        Base Class to plot prediction vs. target
    """

    def __init__(
        self, ax, prediction, target, legend: bool = True, color=None
    ):
        super().__init__(ax)
        self._prediction = prediction
        self._target = target
        self._color = color
        self._legend = legend
        self.plot()

    def plot(self):
        ax = self._ax
        target = self._target
        pred = self._prediction

        x, y_median, y_up, y_down = self.calc_plot_lines()

        scatter_plot = ax.plot(
            target, pred, ls="", marker="o", alpha=0.2, label="Entries"
        )
        if self._color is not None:
            col = self._color
        else:
            col = scatter_plot[0].get_color()

        ax.plot(x, y_median, color=col, lw=2, label="Median")
        ax.plot(
            x,
            y_up,
            color=col,
            ls="dotted",
            lw=2,
            label=r"5% and 95% quantiles",
        )
        ax.plot(x, y_down, color=col, ls="dotted", lw=2)

        ax.plot(x, x, ls="--", color="black")

        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")

        if self._legend:
            ax.legend(frameon=False)

        ax.grid(ls="--", alpha=0.2)

    def calc_plot_lines(self):
        target = self._target
        pred = self._prediction

        xmin = np.min(target)
        xmax = np.max(target)

        ymin = np.min(pred)
        ymax = np.max(pred)

        xbins = np.linspace(xmin, xmax, 20)
        ybins = np.linspace(ymin, ymax, 20)

        hist, xbins, ybins = np.histogram2d(target, pred, bins=(xbins, ybins))

        h = np.cumsum(hist, axis=1)
        norm = h[:, -1]
        h[norm > 0] /= norm[norm > 0][:, np.newaxis]

        xbins_center = (xbins[1:] + xbins[:-1]) / 2.0

        idx_median = np.argmax(h > 0.5, axis=1)
        y_median = ((ybins[1:] + ybins[:-1]) / 2.0)[idx_median]

        idx_up = np.argmax(h > 0.95, axis=1)
        y_up = ((ybins[1:] + ybins[:-1]) / 2.0)[idx_up]

        idx_down = np.argmax(h > 0.05, axis=1)
        y_down = ((ybins[1:] + ybins[:-1]) / 2.0)[idx_down]

        return xbins_center, y_median, y_up, y_down

    @property
    def ax(self):
        """
            Getter function for the ax
        """
        return self._ax
