from simulation.process_models.constant_acceleration import ConstantAcceleration
from simulation.sensors.kalman_sensor_node import KalmanSensorNode

import numpy as np

np.random.seed(42)


class SimpleSensorNetworkSimulation(object):
    def __init__(self, fusion_algorithm):
        self.fusion_algorithm = fusion_algorithm
        self.process = ConstantAcceleration(initial_state=np.zeros(shape=(3,1)))
        self.node_a = KalmanSensorNode(node_id=0, process=self.process, measurement_covariance=np.random.rand(3,3))
        self.node_b = KalmanSensorNode(node_id=1, process=self.process, measurement_covariance=np.random.rand(3,3))

    def run(self, num_timesteps):
        for t in range(num_timesteps):
            self.process.step()
            mean_a, cov_a = self.node_a.estimate()
            mean_b, cov_b = self.node_b.estimate()
            fused_mean, fused_cov = self.fusion_algorithm.fuse(mean_a, cov_a, mean_b, cov_b)
