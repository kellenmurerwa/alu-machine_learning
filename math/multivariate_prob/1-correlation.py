#!/usr/bin/env python3
"""
Defines a function that calculates a correlation matrix
"""


import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d = C.shape[0]
    std_dev = np.sqrt(np.diag(C))
    std_dev_matrix = np.outer(std_dev, std_dev)
    corr = C / std_dev_matrix
    return corr
