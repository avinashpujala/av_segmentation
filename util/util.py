import re
import numpy as np
import time


def apply_recursively(func, x, *args, n_iter=2, **kwargs):
    """
    Apply function recursively
    Parameters
    ----------
    func: object
        Function to apply recursively
    x:
        Parameter to recursively apply funcion to
    args: list-like
        Other arguments of function
    n_iter: int
        # of times to apply to
    kwargs: dict
        Keyword arguments
    Returns
    -------

    """
    count = 0
    while count < n_iter:
        x = func(x, *args, **kwargs)
        count += 1
    return x


def find_str_in_list(match_str, str_list, case_sensitive=False):
    """
    Find a string in a list of strings
    Parameters
    ----------
    match_str: string
        String to search for.
    str_list: list
        List of strings within which to search
    case_sensitive: bool
        If True, then case-sensitive searching.
    Returns
    -------
    inds: array, (n,)
        Indices of items in list (str_list) where the string (match_str) is found.
    """
    inds = []
    for count, sl in enumerate(str_list):
        if case_sensitive:
            matchLen = re.findall(match_str, sl)
        else:
            matchLen = re.findall(match_str.lower(), sl.lower())
        if len(matchLen) > 0:
            inds.append(count)
    return np.array(inds)


def timestamp(till='hour'):
    """
    Returns timestamp (string) till the specified temporal
    resolution
    """
    import time
    if till.lower() == 'year':
        ts = time.strftime('%Y')
    elif till.lower() == 'month':
        ts = time.strftim('%Y%m')
    elif till.lower() == 'day':
        ts = time.strftime('%Y%m%d')
    elif till.lower() == 'hour':
        ts = time.strftime('%Y%m%d-%H')
    elif (till.lower() == 'minute') | (till.lower() == 'min'):
        ts = time.strftime('%Y%m%d-%H%M')
    else:
        ts = time.strftime('%Y%m%d-%H%M%S')
    return ts
