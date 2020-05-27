# -*- coding: utf-8 -*-

import time
from functools import wraps
from pathlib import Path

import pandas_profiling

#from .datahandler import DataHandler

def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print('Function: %r took: %2.2f s' % (f.__name__,  end - start))
        return result
    return wrapper


def generate_report(dataframe,
                    minimal: bool = True,
                    output: bool = None):
    """
        Creates processed_dataframe report using pandas_profiling
    """
    report = pandas_profiling.ProfileReport(
        dataframe, minimal=minimal
    )

    if output is not None:
        directory = Path.cwd().joinpath(output)
        Path(directory).mkdir(parents=True, exist_ok=True)
        report.to_file(
            directory.joinpath("processed_dataframe_report.html")
        )
