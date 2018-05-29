"""
Ellipsoidal intersection according to https://pdfs.semanticscholar.org/30b3/50418ed164f7f5bdb7bf01fec2e9fa0c61d5.pdf
"""

from numpy.linalg import inv, det
import numpy as np
from scipy.linalg import sqrtm


class EllipsoidalIntersection(object):
    def __init__(self):
        self.algorithm_name = "Ellipsoidal Intersection"
        self.algorithm_abbreviation = "EI"

    def fuse(self, mean_a, cov_a, mean_b, cov_b):
        mean_m, cov_m = self.mutual_information(mean_a, cov_a, mean_b, cov_b)
        cov_a_inv = inv(cov_a)
        cov_b_inv = inv(cov_b)
        cov_m_inv = inv(cov_m)
        cov = inv(cov_a_inv + cov_b_inv - cov_m_inv)
        mean = np.dot(cov, np.dot(cov_a_inv, mean_a) + np.dot(cov_b_inv, mean_b) - np.dot(cov_m_inv, mean_m))
        return mean, cov

    def mutual_information(self, mean_a, cov_a, mean_b, cov_b):
        cov_m = self.mutual_covariance(cov_a, cov_b)
        mean = self.mutual_mean(mean_a, cov_a, mean_b, cov_b, cov_m)
        return mean, cov_m

    def mutual_mean(self, mean_a, cov_a, mean_b, cov_b, cov_m):
        dims = mean_a.shape[0]
        cov_m_inv = inv(cov_m)
        cov_a_inv = inv(cov_a)
        cov_b_inv = inv(cov_b)
        H = cov_a_inv + cov_b_inv - np.multiply(2, cov_m_inv)
        if det(H) == 0:
            eta = 0
        else:
            eig_H, _ = np.linalg.eigh(H)
            smallest_nonzero_ev = min(list(filter(lambda x: x != 0, eig_H)))
            eta = 0.0001 * smallest_nonzero_ev
        eta_I = np.multiply(eta, np.identity(dims))
        first_term = inv(cov_a_inv + cov_b_inv - np.multiply(2, cov_m_inv) + np.multiply(2, eta_I))
        second_term = np.dot(cov_b_inv - cov_m_inv + eta_I, mean_a) + np.dot(cov_a_inv - cov_m_inv + eta_I, mean_b)
        return np.dot(first_term, second_term)

    def mutual_covariance(self, cov_a, cov_b):
        D_a, S_a = np.linalg.eigh(cov_a)
        D_a_sqrt = sqrtm(np.diag(D_a))
        D_a_sqrt_inv = inv(D_a_sqrt)
        M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
        D_b, S_b = np.linalg.eigh(M)
        D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
        return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al.