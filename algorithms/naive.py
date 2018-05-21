"""
Naive fusion according to http://discovery.ucl.ac.uk/135541/
"""

from numpy.linalg import inv
import numpy as np


class Naive(object):
    def __init__(self):
        self.algorithm_name = "Naive"
        self.algorithm_abbreviation = "Naive"

    def fuse(self, mean_a, cov_a, mean_b, cov_b):
        cov_a_inv = inv(cov_a)
        cov_b_inv = inv(cov_b)
        cov = inv(cov_a_inv + cov_b_inv)
        K = np.dot(cov, cov_a_inv)
        L = np.dot(cov, cov_b_inv)
        mean = np.dot(K, mean_a) + np.dot(L, mean_b)
        return mean, cov
