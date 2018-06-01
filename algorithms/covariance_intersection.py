"""
Covariance Intersection according to http://discovery.ucl.ac.uk/135541/
"""

from enum import Enum
from scipy.optimize import fminbound
from numpy.linalg import inv, det
import numpy as np


class PerformanceCriterion(Enum):
    TRACE = 1,
    DETERMINANT = 2


class CovarianceIntersection(object):
    def __init__(self, performance_criterion=PerformanceCriterion.TRACE):
        self.performance_criterion = det if performance_criterion == PerformanceCriterion.DETERMINANT else np.trace
        self.algorithm_name = "Covariance Intersection"
        self.algorithm_abbreviation = "CI"

    def fuse(self, mean_a, cov_a, mean_b, cov_b):
        omega = self.optimize_omega(cov_a, cov_b)
        cov = inv(np.multiply(omega, inv(cov_a)) + np.multiply(1 - omega, inv(cov_b)))
        mean = np.dot(cov, (np.dot(np.multiply(omega, inv(cov_a)), mean_a) + np.dot(np.multiply(1 - omega, inv(cov_b)), mean_b)))
        return mean, cov

    def optimize_omega(self, cov_a, cov_b):
        def optimize_fn(omega):
            return self.performance_criterion(inv(np.multiply(omega, inv(cov_a)) + np.multiply(1 - omega, inv(cov_b))))
        return fminbound(optimize_fn, 0, 1)
        #return 0.5
