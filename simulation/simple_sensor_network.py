from simulation.process_models.constant_acceleration import ConstantAcceleration
from simulation.sensors.kalman_sensor_node import KalmanSensorNode

import numpy as np


class SimpleSensorNetworkSimulation(object):
    def __init__(self, fusion_algorithm):
        self.process = ConstantAcceleration(initial_state=np.zeros(shape=(3,1)), delta_t=0.01)
        random_matrix_A = np.random.rand(3,3)
        random_matrix_B = np.random.rand(3,3)
        measurement_cov_a = np.dot(random_matrix_A, random_matrix_A.T)
        measurement_cov_b = np.dot(random_matrix_B, random_matrix_B.T)
        self.node_a = KalmanSensorNode(0, self.process, fusion_algorithm, transition_matrices=self.process.F, measurement_covariance=measurement_cov_a)
        self.node_b = KalmanSensorNode(1, self.process, fusion_algorithm, transition_matrices=self.process.F, measurement_covariance=measurement_cov_b)

    def run(self, num_timesteps):
        for t in range(num_timesteps):
            self.process.step()
            mean_a, cov_a = self.node_a.estimate()
            mean_b, cov_b = self.node_b.estimate()
            self.node_a.fuse_in(mean_b, cov_b)
            self.node_b.fuse_in(mean_a, cov_a)
