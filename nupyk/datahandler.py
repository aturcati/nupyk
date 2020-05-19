# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .sedreader import SEDReader


class BaseHandler(object, metaclass=ABCMeta):
    """
        Base class for SED data
    """

    def __init__(
        self, input_directory: [list, str], output_directory: str = None
    ) -> None:
        if not isinstance(input_directory, (list, tuple)):
            input_directory = [input_directory]
        self._input_dir = input_directory
        self._output_dir = output_directory

    @abstractmethod
    def read(self):
        """
            Read raw data.
        """

    @abstractmethod
    def process(self):
        """
            Process raw data and creates dataframe
            with all the necessary features.
        """

    @abstractmethod
    def save(self):
        """
            Save processed data.
        """


class DataReader(BaseHandler):
    """
        Base class to read SED data.
        - input_directory  : path that contains the SED data files
        - output_directory : path where the output can be saved
    """

    def __init__(
        self, input_directory: [list, str], output_directory: str = None
    ) -> None:
        super().__init__(input_directory, output_directory)
        print("Reading files...")
        self.read()

    def read(self):
        """
            Read SED data in the format provided by http://www.openuniverse.asi.it/
            Creates a pandas DataFrame 'raw_dataframe'. Each row contains information
            about the source:
            - 'name'    : name (str)
            - 'dec'     : declination (float)
            - 'ra'      : right ascension (float)
            - 'nu_peak' : position of the nu peak (float)
            - 'redshift': redshift (if available) (float)
            - 'data'    : pandas DataFrame from SEDReader

        """
        sed_files = self.get_files_paths()

        sources_data_list = list()
        for i, filepath in enumerate(tqdm(sed_files[:10])):
            filename = os.path.basename(os.path.normpath(filepath)).split("_")
            source_name = filename[4].replace(".", "")
            redshift = np.float(filename[5]) if len(filename) == 6 else np.nan
            try:
                sources_data_list.append(
                    {
                        "name": source_name,
                        "dec": np.float(filename[3]),
                        "ra": np.float(filename[2]),
                        "nu_peak": np.float(filename[1]),
                        "redshift": redshift,
                        "data": SEDReader(filepath).dataframe,
                    }
                )
            except:
                print("Error reading file: {0}".format(filepath))

        sources_dataframe = pd.DataFrame(sources_data_list)
        self._raw_dataframe = sources_dataframe

    def get_files_paths(self) -> list:
        """
            Creates a list that contains the paths of the SED data files
        """
        sed_files_paths = list()
        for indir in self._input_dir:
            for dirpath, _, filenames in os.walk(indir):
                for f in filenames:
                    sed_files_paths.append(
                        os.path.abspath(os.path.join(dirpath, f))
                    )
        return sed_files_paths

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        """
            Getter function for the raw dataframe
        """
        return self._raw_dataframe

    def save(self, filename: str = None):
        """
            Functon to save the raw_dataframe
        """
        if self._output_dir is not None:
            directory = self._output_dir
        else:
            directory = Path.cwd().joinpath("output")

        Path(directory).mkdir(parents=True, exist_ok=True)

        df = self._raw_dataframe
        if filename is not None:
            df.to_pickle(directory.joinpath(filename + ".pkl"))
        else:
            df.to_pickle(directory.joinpath("raw_dataframe.pkl"))


class DataHandler(DataReader):
    """
        Base class to read and process SED data.

        - input_directory  : path that contains the SED data files
        - output_directory : path where the output can be saved
        - agg_func         : list of functions that will be applied to the
                             pandas grouped dataframe
        - freq_bins        : list of the frequency bin margins to group
                             the dataframe
    """

    def __init__(
        self,
        input_directory: [list, str],
        output_directory: str = None,
        **kwargs
    ) -> None:
        super().__init__(input_directory, output_directory)

        agg_func_list = kwargs.get("agg_func")
        if agg_func_list == None:
            agg_func_list = ["mean", "std", "min", "max", "count"]
        self._agg_func_list = agg_func_list

        freq_bins = kwargs.get("freq_bins")
        if freq_bins == None:
            freq_bins = np.array([1e8, 3e12, 2.4e14, 1e15, 2e20, 1e35])
            freq_bins_names = ["radio", "IR", "optical", "X", "gamma"]
            feature_names = list()
            for name in freq_bins_names:
                feature_names.append(
                    "{0}_freq_{1}".format(name, agg_func_list[0])
                )
                for func in agg_func_list:
                    feature_names.append("{0}_flux_{1}".format(name, func))

        else:
            freq_bins = np.array(freq_bins)
            feature_names = list()
            freq_bins_names = list()
            for name in range(len(freq_bins)):
                freq_bins_names.append("{0}".format(name))
                feature_names.append(
                    "{0}_freq_{1}".format(name, agg_func_list[0])
                )
                for func in agg_func_list:
                    feature_names.append("{0}_flux_{1}".format(name, func))

        self._frequency_bins = freq_bins
        self._frequency_bins_names = freq_bins_names
        self._feature_names = feature_names

        processed_dataframe = self.raw_dataframe.drop("data", axis=1)
        processed_dataframe = processed_dataframe.reindex(
            processed_dataframe.columns.tolist() + self._feature_names, axis=1
        )
        self._processed_dataframe = processed_dataframe
        print("Processing data...")
        self.process()

    def process(self):
        """
            Process the raw_dataframe to a processed_dataframe that contains
            the requested features for each source.
        """
        data = self.raw_dataframe

        proc_df = self.processed_dataframe

        for i, sed_data in enumerate(tqdm(data["data"])):
            sed_data["freq_bins_index"] = self.calculate_freq_index(sed_data)
            freq_bin_group = (
                sed_data.drop(["flux_min", "flux_plus"], axis=1)
                .groupby("freq_bins_index")
                .agg(self._agg_func_list)
            )
            for name in self._frequency_bins_names:
                for func in self._agg_func_list:
                    try:
                        proc_df.loc[
                            i, "{0}_freq_{1}".format(name, func)
                        ] = freq_bin_group["freq"][func][name]
                    except:
                        proc_df.loc[
                            i, "{0}_freq_{1}".format(name, func)
                        ] = np.nan

                    try:
                        proc_df.loc[
                            i, "{0}_flux_{1}".format(name, func)
                        ] = freq_bin_group["flux"][func][name]
                    except:
                        proc_df.loc[
                            i, "{0}_flux_{1}".format(name, func)
                        ] = np.nan

        self._processed_dataframe = proc_df

    def calculate_freq_index(self, data: pd.DataFrame) -> list:
        """
            Calculate the index of the frequency bin for each observation
            of the source contained in the dataframe provided by the SEDReader.
            Returns a list of the indexes.
        """
        frequency = data["freq"].to_numpy()
        freq_bins_index = np.asarray(
            (
                np.sum(
                    self._frequency_bins <= frequency[:, np.newaxis], axis=1
                )
                - 1
            ),
            dtype=np.int,
        )
        return [self._frequency_bins_names[i] for i in freq_bins_index]

    @property
    def frequency_bins(self) -> np.array:
        """
            Getter function for the frequency bins
        """
        return self._frequency_bins

    @property
    def frequency_bins_names(self) -> list:
        """
            Getter function for the frequency bins names
        """
        return self._frequency_bins_names

    @property
    def feature_names(self) -> list:
        """
            Getter function for the feature names
        """
        return self._feature_names

    @property
    def processed_dataframe(self) -> pd.DataFrame:
        """
            Getter function for the processed dataframe
        """
        return self._processed_dataframe

    def save(self, filename: str = None):
        """
            Functon to save the processed_dataframe
        """
        if self._output_dir is not None:
            directory = self._output_dir
        else:
            directory = Path.cwd().joinpath("output")

        Path(directory).mkdir(parents=True, exist_ok=True)

        df = self._processed_dataframe
        if filename is not None:
            df.to_pickle(directory.joinpath(filename + ".pkl"))
        else:
            df.to_pickle(directory.joinpath("processed_dataframe.pkl"))

    def save_raw(self, filename: str = None):
        """
            Functon to save the raw_dataframe
        """
        super().save(filename)
