from simulation.process_models.constant_acceleration import ConstantAcceleration
from simulation.sensors.kalman_sensor_node import KalmanSensorNode

import numpy as np


class SimpleSensorNetworkSimulation(object):
    def __init__(self, fusion_algorithms):
        self.fusion_algorithms = fusion_algorithms
        self.process = ConstantAcceleration(initial_state=np.zeros(shape=(3,1)), delta_t=0.01)
        random_matrix_A = np.random.rand(3,3)
        random_matrix_B = np.random.rand(3,3)
        measurement_cov_a = np.dot(random_matrix_A, random_matrix_A.T)
        measurement_cov_b = np.dot(random_matrix_B, random_matrix_B.T)
        self.node_a = KalmanSensorNode(node_id=0, process=self.process, transition_matrices=self.process.F, measurement_covariance=measurement_cov_a)
        self.node_b = KalmanSensorNode(node_id=1, process=self.process, transition_matrices=self.process.F, measurement_covariance=measurement_cov_b)

    def run(self, num_timesteps):
        for t in range(num_timesteps):
            self.process.step()
            mean_a, cov_a = self.node_a.estimate()
            mean_b, cov_b = self.node_b.estimate()
            fusion_results = []
            for fusion_algorithm in self.fusion_algorithms:
                fusion_results.append(fusion_algorithm.fuse(mean_a, cov_a, mean_b, cov_b))
            yield fusion_results
