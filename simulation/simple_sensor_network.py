from simulation.sensors.kalman_sensor_node import KalmanSensorNode

import numpy as np


class SimpleSensorNetworkSimulation(object):
    def __init__(self, fusion_algorithm, process, sensor_nodes=None):
        self.process = process
        if sensor_nodes is not None:
            self.node_a = sensor_nodes[0]
            self.node_b = sensor_nodes[1]
        else:
            random_matrix_A = np.random.rand(2, 2) * 0.3
            random_matrix_B = np.random.rand(2, 2) * 0.3
            measurement_cov_a = np.dot(random_matrix_A, random_matrix_A.T)
            measurement_cov_b = np.dot(random_matrix_B, random_matrix_B.T)
            self.node_a = KalmanSensorNode(0, 2, self.process, fusion_algorithm, transition_matrix=self.process.F,
                             transition_covariance=self.process.Q, measurement_covariance=measurement_cov_a)
            self.node_b = KalmanSensorNode(1, 2, self.process, fusion_algorithm, transition_matrix=self.process.F,
                             transition_covariance=self.process.Q, measurement_covariance=measurement_cov_b)

    def run(self, num_timesteps):
        for t in range(num_timesteps):
            self.process.step()
            mean_a, cov_a = self.node_a.estimate()
            mean_b, cov_b = self.node_b.estimate()
            self.node_a.fuse_in(mean_b, cov_b)
            self.node_b.fuse_in(mean_a, cov_a)
