# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import os

import numpy as np
import pandas as pd


class BaseReader(object, metaclass=ABCMeta):
    """
        Base class to read Spectral Energy Distribution data
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self.read()

    @abstractmethod
    def read(self):
        """
            Read raw data.
        """
        self._dataframe = None

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        self._dataframe = df


class SEDReader(BaseReader):
    """
        Read SED data in the format provided by http://www.openuniverse.asi.it/
        Provides a pandas dataframe that for each row (measurement) contains
        - freq      : frequency of the observation
        - flux      : flux measured
        - flux_plus : flux upper limit
        - flux_min  : flux lower limit
        - cat       : catalog
        If the measurements are upper limits they are removed.
    """

    def read(self):
        f = np.genfromtxt(
            self._filepath,
            usecols=[0, 1, 2, 3, 6],
            skip_header=4,
            dtype=[
                ("freq", np.float),
                ("flux", np.float),
                ("flux_plus", np.float),
                ("flux_min", np.float),
                ("cat", "U8"),
            ],
        )
        df = pd.DataFrame(f)
        df["uplim"] = pd.Series(
            np.logical_and(
                df["flux"] == df["flux_plus"], df["flux"] == df["flux_min"]
            )
        )
        df = df[df.uplim == False]
        self._dataframe = df.drop("uplim", axis=1)
