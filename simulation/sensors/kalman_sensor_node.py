"""
A simulated sensor node. Each node runs its own, local Kalman filter to estimate the state.
"""

from pykalman import KalmanFilter
import numpy as np


class KalmanSensorNode(object):
    def __init__(self, node_id, process, fusion_algorithm, transition_matrices, measurement_covariance):
        self.node_id = node_id
        self.mean = np.zeros(shape=(3,))
        self.cov = np.identity(3)
        self.kf = KalmanFilter(transition_matrices=transition_matrices, observation_covariance=measurement_covariance)
        self.process = process
        self.fusion_algorithm = fusion_algorithm
        self.local_estimates = []
        self.fused_estimates = []

    def estimate(self):
        self.mean, self.cov = self.kf.filter_update(self.mean, self.cov, np.squeeze(np.asarray(self.process.current_state)))
        self.local_estimates.append((self.mean, self.cov))
        return self.mean, self.cov

    def fuse_in(self, mean, cov):
        self.mean, self.cov = self.fusion_algorithm.fuse(self.mean, self.cov, mean, cov)
        self.fused_estimates.append((self.mean, self.cov))
        return self.mean, self.cov
