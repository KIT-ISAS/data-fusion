"""
A simulated sensor node. Each node runs its own, local Kalman filter to estimate the state.
"""

from pykalman import KalmanFilter
import numpy as np


class KalmanSensorNode(object):
    def __init__(self, node_id, process, measurement_covariance):
        self.node_id = node_id
        self.mean = 0
        self.cov = np.identity(3)
        self.kf = KalmanFilter(observation_covariance=measurement_covariance)
        self.process = process

    def estimate(self):
        self.mean, self.cov = self.kf.filter_update(self.mean, self.cov, self.process.current_state)
        return self.mean, self.cov
