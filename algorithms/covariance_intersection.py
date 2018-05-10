from enum import Enum
from scipy.optimize import minimize
from numpy.linalg import inv, det
import numpy as np


class PerformanceCriterion(Enum):
    TRACE = 1,
    DETERMINANT = 2


class CovarianceIntersection(object):
    def __init__(self, performance_criterion=PerformanceCriterion.DETERMINANT):
        self.performance_criterion = det if performance_criterion == PerformanceCriterion.DETERMINANT else np.trace

    def fuse(self, mean_a, cov_a, mean_b, cov_b):
        omega = self.optimize_omega(cov_a, cov_b)
        p_cc = inv(np.multiply(omega, inv(cov_a)) + np.multiply(1 - omega, inv(cov_b)))
        c = np.dot(p_cc, (np.dot(np.multiply(omega, inv(cov_a)), mean_a) + np.dot(np.multiply(1 - omega, inv(cov_b)), mean_b)))
        return c, p_cc

    def optimize_omega(self, cov_a, cov_b):
        def optimize_fn(omega):
            return self.performance_criterion(inv(np.multiply(omega, inv(cov_a)) + np.multiply(1 - omega, inv(cov_b))))
        omega_0 = np.array([1.0])
        return minimize(optimize_fn, omega_0).x[0]