"""
A simulated sensor node. Each node runs its own, local Kalman filter to estimate the state.
"""

from pykalman import KalmanFilter
import numpy as np


class KalmanSensorNode(object):
    def __init__(self, node_id, state_dims, process, fusion_algorithm, transition_matrix, transition_covariance, measurement_covariance):
        self.node_id = node_id
        self.state_dims = state_dims
        self.measurement_covariance = measurement_covariance
        self.mean = np.zeros(shape=(state_dims,))
        self.cov = np.identity(state_dims)
        self.kf = KalmanFilter(transition_matrices=transition_matrix, transition_covariance=transition_covariance, observation_covariance=measurement_covariance)
        self.process = process
        self.fusion_algorithm = fusion_algorithm
        self.local_estimates = []
        self.fused_estimates = []

    def estimate(self):
        measurement_noise = np.random.multivariate_normal([0 for dim in range(self.state_dims)], self.measurement_covariance)
        noisy_measurement = np.squeeze(np.asarray(self.process.current_state)) + measurement_noise
        self.mean, self.cov = self.kf.filter_update(self.mean, self.cov, noisy_measurement)
        self.local_estimates.append((self.mean, self.cov))
        return self.mean, self.cov

    def fuse_in(self, mean, cov):
        self.mean, self.cov = self.fusion_algorithm.fuse(self.mean, self.cov, mean, cov)
        self.fused_estimates.append((self.mean, self.cov))
        return self.mean, self.cov
