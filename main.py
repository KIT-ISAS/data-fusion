from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.naive import Naive

import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    seed = random.randint(0, 1000)
    print("Seed: {}".format(seed))
    np.random.seed(seed)
    fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
    sim = SimpleSensorNetworkSimulation(fusion_algorithms)
    sim_results = list(sim.run(50)) # List of list of tuples (mean, covariance)
    process_states = list(map(lambda x: np.squeeze(np.asarray(x)), sim.process.states))

    # Plot results
    plt.rcParams["figure.figsize"] = (10, 10)
    pos_real = [x[0] for x in process_states]
    vel_real = [x[1] for x in process_states]
    acc_real = [x[2] for x in process_states]
    pos_a = [x[0] for x in sim.node_a.means]
    vel_a = [x[1] for x in sim.node_a.means]
    acc_a = [x[2] for x in sim.node_a.means]
    pos_fused = []
    vel_fused = []
    acc_fused = []
    for i in range(len(fusion_algorithms)):
        pos_fused.append([x[i][0][0] for x in sim_results])
        vel_fused.append([x[i][0][1] for x in sim_results])
        acc_fused.append([x[i][0][2] for x in sim_results])

    plt.subplot(3, 1, 1)
    plt.plot(pos_real, label="Real")
    plt.plot(pos_a, label="Node A")
    for i in range(len(fusion_algorithms)):
        plt.plot(pos_fused[i], label=fusion_algorithms[i].algorithm_abbreviation)
    plt.legend()
    plt.title("Position")

    plt.subplot(3, 1, 2)
    plt.plot(vel_real, label="Real")
    plt.plot(vel_a, label="Node A")
    for i in range(len(fusion_algorithms)):
        plt.plot(vel_fused[i], label=fusion_algorithms[i].algorithm_abbreviation)
    plt.legend()
    plt.title("Velocity")

    plt.subplot(3, 1, 3)
    plt.plot(acc_real, label="Real")
    plt.plot(acc_a, label="Node A")
    for i in range(len(fusion_algorithms)):
        plt.plot(acc_fused[i], label=fusion_algorithms[i].algorithm_abbreviation)
    plt.legend()
    plt.title("Acceleration")

    plt.show()

if __name__ == "__main__":
    main()