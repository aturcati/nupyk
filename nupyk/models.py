from abc import ABCMeta, abstractmethod

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pickle

import xgboost as xgb
from xgboost.sklearn import XGBRegressor


class BaseTrainer(metaclass=ABCMeta):
    """
        Base classe to define a model trainer
    """

    def __init__(self,
                 data: [str, pd.DataFrame] = None,
                 directory: str = None,
                 model: XGBRegressor = None
        ):
        """
            directory : path in which the defined model will be saved
        """
        if directory is not None:
            self._output_dir = directory
        else:
            self._output_dir = Path.cwd().joinpath("models")

        if isinstance(data, str):
            self.load_pickle_data(filepath=data)
        else:
            self._dataframe = data

        if model is not None:
            self._model = model
        else:
            self.def_model()

    def load_pickle_data(self, filepath: str):
        """
            load dataframe from pickled file
        """
        dataframe = pd.read_pickle(filepath)
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        """
            Getter method for the data
        """
        return self._dataframe

    @dataframe.setter
    def dataframe(self, data: pd.DataFrame):
        """
            Setter method for the data
        """
        self._dataframe = data

    @abstractmethod
    def process(self):
        """
            Prepare the data for training
        """

    @abstractmethod
    def def_model(self):
        """
            Define the model
        """

    def load_pickle_model(self, filepath: str):
        """
            load model from pickled file
        """
        model = pickle.load(open(filepath))
        self._model = model

    @property
    def model(self):
        """
            Getter method for the model
        """
        return self._model

    @model.setter
    def model(self, model):
        """
            Setter method for the model
        """
        self._model = model

    @abstractmethod
    def fit_model(self):
        """
            Train the model
        """

    @abstractmethod
    def generate_metrics(self):
        """
            Generate metrics using trained model and test data.
        """

    @abstractmethod
    def save_model(self, model_name):
        """
            This method saves the model in our required format.
        """


class XGBTrainer(BaseTrainer):
    """
        Trainer based on XGBoost library
    """

    def def_model(self, **kwargs):
        model = XGBRegressor(**kwargs)
        self._model = model

    def fit_model(self):
        pass

    def generate_metrics(self):
        pass

    def process(self):
        pass

    def save_model(self, filename: str = None):
        directory = self._output_dir

        Path(directory).mkdir(parents=True, exist_ok=True)

        model = self._model
        if filename is not None:
            pickle.dump(model, open(directory.joinpath(filename + ".pkl"), "wb"))
        else:
            pickle.dump(model, open(directory.joinpath("xgb_model.pkl"), "wb"))
