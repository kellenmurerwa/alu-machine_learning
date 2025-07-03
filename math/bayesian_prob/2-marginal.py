#!/usr/bin/env python3
"""
Defines a function that calculates the marginal probability
of obtaining the data
"""


import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data

    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")
    factorial = np.math.factorial
    factorialCoefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = factorialCoefficient * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    marginal = np.sum(intersection)
    return marginal
