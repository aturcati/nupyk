# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pickle

import xgboost as xgb
from xgboost.sklearn import XGBRFRegressor, XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
)

from .utils import timing


class BaseTrainer(metaclass=ABCMeta):
    """
        Base classe to define a model trainer
    """

    def __init__(
        self,
        data: [str, pd.DataFrame] = None,
        directory: str = None,
        model=None,
        random_state=0,
    ):
        self._random_state = random_state

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
        filepath = Path(filepath)
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
        filepath = Path(filepath)
        model = pickle.load(open(filepath, "rb"))
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


class XGBRTrainer(BaseTrainer):
    """
        Trainer based on XGBRegressor from XGBoost library
    """

    def process(
        self,
        target="nu_peak",
        ignored_features=["name", "nu_peak", "redshift"],
        train_size=0.8,
    ):
        self._target_name = target
        self._ignored_features_names = ignored_features
        self._feature_names = sorted([
            feat_name
            for feat_name in self._dataframe.columns
            if (
                (feat_name not in self._ignored_features_names)
                and (feat_name not in self._target_name)
            )
        ])

        x_train, x_test, y_train, y_test = train_test_split(
            self._dataframe[self._feature_names],
            self._dataframe[self._target_name],
            train_size=train_size,
            random_state=self._random_state,
        )

        self._x_train = x_train
        self._x_test = x_test
        self._target_train = y_train
        self._target_test = y_test

    @property
    def target_name(self):
        return self._target_name

    @property
    def ignored_features_names(self):
        return self._ignored_features_names

    @property
    def feature_names(self):
        return self._feature_names

    def def_model(self, parameters: dict = None):
        model = XGBRegressor()

        if parameters is not None:
            model.set_params(**parameters)

        self._model = model

    @timing
    def fit_model(self):
        model = self._model
        model.fit(self._x_train, self._target_train)

    def generate_metrics(self):
        model = self.model
        target = self._target_test
        prediction = model.predict(self._x_test)

        met_dict = {
            'explained_variance_score': explained_variance_score(target, prediction),
            'max_error': max_error(target, prediction),
            'mean_absolute_error': mean_absolute_error(target, prediction),
            'mean_squared_error': mean_squared_error(target, prediction),
            'mean_squared_log_error': mean_squared_log_error(target, prediction),
            'median_absolute_error': median_absolute_error(target, prediction),
            'r2_score': r2_score(target, prediction),
            'mean_poisson_deviance': mean_poisson_deviance(target, prediction),
            'mean_gamma_deviance': mean_gamma_deviance(target, prediction)
        }

        self._model_metrics = pd.DataFrame.from_dict(met_dict, orient='index')

    @property
    def model_metrics(self):
        return self._model_metrics

    def save_metrics(self, filename: str = None):
        directory = self._output_dir

        Path(directory).mkdir(parents=True, exist_ok=True)

        metrics = self._model_metrics

        if filename is not None:
            metrics.to_csv(directory.joinpath(filename + ".metrics"))
        else:
            metrics.to_csv(directory.joinpath("xgb_model" + ".metrics"))

    def save_model(self, filename: str = None):
        directory = self._output_dir

        Path(directory).mkdir(parents=True, exist_ok=True)

        model = self._model
        if filename is not None:
            pickle.dump(
                model, open(directory.joinpath(filename + ".pkl"), "wb")
            )
        else:
            pickle.dump(model, open(directory.joinpath("xgb_model.pkl"), "wb"))


class XGBRFTrainer(XGBRTrainer):
    """
        Trainer based on XGBRFRegressor from XGBoost library
    """

    def def_model(self, parameters: dict = None):
        model = XGBRFRegressor()

        if parameters is not None:
            model.set_params(**parameters)

        self._model = model
