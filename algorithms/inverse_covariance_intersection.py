"""
Inverse covariance intersection according to http://discovery.ucl.ac.uk/135541/
"""

from enum import Enum
from scipy.optimize import fminbound
from numpy.linalg import inv, det
import numpy as np


class PerformanceCriterion(Enum):
    TRACE = 1,
    DETERMINANT = 2


class InverseCovarianceIntersection(object):
    def __init__(self, performance_criterion=PerformanceCriterion.TRACE):
        self.performance_criterion = det if performance_criterion == PerformanceCriterion.DETERMINANT else np.trace
        self.algorithm_name = "Inverse Covariance Intersection"
        self.algorithm_abbreviation = "ICI"

    def fuse(self, mean_a, cov_a, mean_b, cov_b):
        omega = self.optimize_omega(cov_a, cov_b)
        cov = inv(inv(cov_a) + inv(cov_b) - inv(np.multiply(omega, cov_a) + np.multiply(1 - omega, cov_b)))
        T = inv(np.multiply(omega, cov_a) + np.multiply(1-omega, cov_b))
        K = np.dot(cov, inv(cov_a) - np.multiply(omega, T))
        L = np.dot(cov, inv(cov_b) - np.multiply(1 - omega, T))
        mean = np.dot(K, mean_a) + np.dot(L, mean_b)
        return mean, cov

    def optimize_omega(self, cov_a, cov_b):
        def optimize_fn(omega):
            return self.performance_criterion(inv(np.multiply(omega, inv(cov_a)) + np.multiply(1 - omega, inv(cov_b))))
        return fminbound(optimize_fn, 0, 1)
