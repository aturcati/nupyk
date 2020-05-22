# -*- coding: utf-8 -*-

import time
from functools import wraps


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
