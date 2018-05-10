from simulation.process_models.constant_acceleration import ConstantAcceleration
from simulation.sensors.kalman_sensor_node import KalmanSensorNode
from algorithms.covariance_intersection import CovarianceIntersection

import numpy as np
import matplotlib.pyplot as plt


class SimpleSensorNetworkSimulation(object):
    def __init__(self, fusion_algorithm):
        self.fusion_algorithm = fusion_algorithm
        self.process = ConstantAcceleration(initial_state=np.zeros(shape=(3,1)), delta_t=0.01)
        self.node_a = KalmanSensorNode(node_id=0, process=self.process, transition_matrices=self.process.F, measurement_covariance=np.random.rand(3,3))
        self.node_b = KalmanSensorNode(node_id=1, process=self.process, transition_matrices=self.process.F, measurement_covariance=np.random.rand(3,3))

    def run(self, num_timesteps):
        for t in range(num_timesteps):
            self.process.step()
            mean_a, cov_a = self.node_a.estimate()
            mean_b, cov_b = self.node_b.estimate()
            yield self.fusion_algorithm.fuse(mean_a, cov_a, mean_b, cov_b)


def main():
    fusion_algorithm = CovarianceIntersection()
    sim = SimpleSensorNetworkSimulation(fusion_algorithm)
    sim_results = list(sim.run(50))
    process_states = list(map(lambda x: np.squeeze(np.asarray(x)), sim.process.states))

    # Plot results
    plt.rcParams["figure.figsize"] = (10, 10)
    pos_real = [x[0] for x in process_states]
    vel_real = [x[1] for x in process_states]
    acc_real = [x[2] for x in process_states]
    pos_a = [x[0] for x in sim.node_a.means]
    vel_a = [x[1] for x in sim.node_a.means]
    acc_a = [x[2] for x in sim.node_a.means]
    pos_fused = [x[0][0] for x in sim_results]
    vel_fused = [x[0][1] for x in sim_results]
    acc_fused = [x[0][2] for x in sim_results]

    print("Process states: {}".format(len(pos_real)))
    print("Estimates: {}".format(len(pos_a)))
    print("Fusion results: {}".format(len(pos_fused)))

    plt.subplot(3, 1, 1)
    plt.plot(pos_real, label="Real", linestyle="solid", color="blue")
    plt.plot(pos_a, label="A", linestyle="dashed", color="blue")
    plt.plot(pos_fused, label="Fused", linestyle="dotted", color="blue")
    plt.legend()
    plt.title("Position")

    plt.subplot(3, 1, 2)
    plt.plot(vel_real, label="Real", linestyle="solid", color="green")
    plt.plot(vel_a, label="A", linestyle="dashed", color="green")
    plt.plot(vel_fused, label="Fused", linestyle="dotted", color="green")
    plt.legend()
    plt.title("Velocity")

    plt.subplot(3, 1, 3)
    plt.plot(acc_real, label="Real", linestyle="solid", color="red")
    plt.plot(acc_a, label="A", linestyle="dashed", color="red")
    plt.plot(acc_fused, label="Fused", linestyle="dotted", color="red")
    plt.legend()
    plt.title("Acceleration")

    plt.show()


if __name__ == "__main__":
    main()
