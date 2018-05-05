from estimate import Estimate
from enum import Enum
from scipy.optimize import minimize
from numpy.linalg import inv, det
import numpy as np

class PerformanceCriterion(Enum):
    TRACE = 1,
    DETERMINANT = 2


class CovarianceIntersection(object):
    def __init__(self, performance_criterion):
        self.performance_criterion = det if performance_criterion == PerformanceCriterion.DETERMINANT else np.trace

    def fuse(self, a, b):
        omega = self.optimize_omega(a, b)
        p_cc = inv(omega * inv(a.cov) + (1 - omega) * inv(b.cov))
        c = p_cc * (omega * inv(a.cov) * a + (1 - omega) * inv(b.cov) * b)
        return Estimate(c, p_cc)

    def optimize_omega(self, a, b):

        def optimize_fn(omega):
            self.performance_criterion(inv(omega * inv(a.cov) + (1 - omega) * inv(b.cov)))
        omega_0 = np.array([1.0])
        return minimize(optimize_fn, omega_0)[0]
