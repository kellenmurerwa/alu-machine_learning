#!/usr/bin/env python3
"""
Defines class MultiNormal that represents a Multivariate Normal Distribution
"""


import numpy as np


class MultiNormal:
    """
    Class that represents Multivariate Normal Distribution
    """

    def __init__(self, data):
        """
        Class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        return pdf[0][0]#!/usr/bin/env python3
"""
Multivariate Normal Distribution Class
"""

import numpy as np

class MultiNormal:
    """
    Represents a Multivariate Normal Distribution.
    """

    def __init__(self, data):
        """
        Initialize MultiNormal with data.
        """
        # Input validation
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        # Store dimensions
        self.d, self.n = data.shape
        # Calculate mean (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)
        # Center the data
        centered = data - self.mean
        # Calculate covariance matrix (d, d)
        self.cov = (centered @ centered.T) / (self.n - 1)
    def __str__(self):
        """String representation of the distribution"""
        return f"MultiNormal(mean={self.mean.T}, cov={self.cov})"
    def pdf(self, x):
        """
        Calculate probability density function at point x.
        """
        # Input validation
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have shape ({self.d}, 1)")
        # Calculate PDF
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        exponent = -0.5 * (x - self.mean).T  (x - self.mean)
        coefficient = 1 / ((2 * np.pi) ** (self.d / 2) * np.sqrt(det))
        return float(coefficient * np.exp(exponent))
    