# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from .sedreader import SEDReader

colors = sns.color_palette("colorblind", 10)

class BaseSourcePlot(object, metaclass=ABCMeta):
    """
        Base Class to plot source data
    """
    def __init__(self, ax, source_obj):
        self._ax = ax
        self._source_obj = source_obj
        self.plot_source()

    @abstractmethod
    def plot_source(self):
        pass

    @property
    def ax(self):
        """
            Getter function for the ax
        """
        return self._ax

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


    def plot_source(self):
        df = self.source_obj.dataframe
        self._ax.errorbar(
            df["freq"],
            df["flux"],
            yerr=[df["flux_plus"] - df["flux"], df["flux"] - df["flux_min"]],
            ls="",
            marker=".",
            color=colors[0],
            alpha=0.2,
            label='Raw Data'
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

    def plot_source(self):
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
                color=colors[i],
                label="Proc. Data ({0})".format(xn.split('_')[0])
            )

        self._ax.set_xlim([1e8, 1e28])
        self._ax.set_ylim([1e-15, 1e-8])
        self._ax.set_title(source['name'])
        self._ax.set_xlabel("Frequency [Hz]")
        self._ax.set_ylabel("nuFnu [erg/cm2/s]")
        self._ax.loglog()
