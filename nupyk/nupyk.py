from abc import ABCMeta, abstractmethod

import numpy as np

from .datahandler import DataHandler
from .trainers import XGBRTrainer


class BasePredictor(metaclass=ABCMeta):
    def __init__(
        self, data: tuple = None, model: object = None,
    ):
        self._source_name = np.atleast_1d(data[0])
        self._source_ra = np.atleast_1d(data[1])
        self._source_dec = np.atleast_1d(data[2])
        self._data = data[3]

        if model is not None:
            self._model = model
        else:
            self._model = None

        self._prediction = None

    @property
    def model(self):
        return self._model

    @property
    def prediction(self):
        return self._prediction

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class XGBRPredictor(BasePredictor):
    def __init__(
        self, data: tuple = None, model: object = None, verbose: bool = False
    ):
        super().__init__(data, model)
        self.process()
        self.predict(verbose)

    def process(self):
        processed_dataframe = DataHandler(
            input_directory=self._data, training_mode=False
        ).processed_dataframe

        processed_dataframe["ra"] = self._source_ra
        processed_dataframe["dec"] = self._source_dec

        self._processed_data = processed_dataframe[
            sorted(processed_dataframe.columns)
        ]

    def predict(self, verbose: bool):
        model = self._model
        processed_data = self._processed_data.drop("name", axis=1)
        prediction = model.predict(processed_data)
        if verbose:
            for sn, pred in zip(self._source_name, prediction):
                print(f"Source {sn} - Nu Peak {pred}")
        self._prediction = prediction
