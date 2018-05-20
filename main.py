from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.naive import Naive

import matplotlib.pyplot as plt
import numpy as np
import random
import argparse


def main(args):
    seed = random.randint(0, 1000) if args.seed is None else args.seed
    print("Seed: {}".format(seed))
    np.random.seed(seed)
    fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
    sim = SimpleSensorNetworkSimulation(fusion_algorithms)
    sim_results = list(sim.run(50)) # List of list of tuples (mean, covariance)
    process_states = list(map(lambda x: np.squeeze(np.asarray(x)), sim.process.states))

    # Plot process
    plt.rcParams["figure.figsize"] = (4, 3)
    proc_fig = plt.figure(1)
    proc_axes = proc_fig.add_subplot(111)
    sim.process.plot(proc_axes)
    proc_fig.show()

    # Plot results
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

    plt.rcParams["figure.figsize"] = (8, 8)
    res_fig = plt.figure(2)
    pos_axes = res_fig.add_subplot(3, 1, 1)
    pos_axes.plot(pos_real, label="Real")
    pos_axes.plot(pos_a, label="Node A")
    for i in range(len(fusion_algorithms)):
        pos_axes.plot(pos_fused[i], label=fusion_algorithms[i].algorithm_abbreviation)
    pos_axes.legend()
    pos_axes.set_title("Position")

    vel_axes = res_fig.add_subplot(3, 1, 2)
    vel_axes.plot(vel_real, label="Real")
    vel_axes.plot(vel_a, label="Node A")
    for i in range(len(fusion_algorithms)):
        vel_axes.plot(vel_fused[i], label=fusion_algorithms[i].algorithm_abbreviation)
    vel_axes.legend()
    vel_axes.set_title("Velocity")

    acc_axes = res_fig.add_subplot(3, 1, 3)
    acc_axes.plot(acc_real, label="Real")
    acc_axes.plot(acc_a, label="Node A")
    for i in range(len(fusion_algorithms)):
        acc_axes.plot(acc_fused[i], label=fusion_algorithms[i].algorithm_abbreviation)
    acc_axes.legend()
    acc_axes.set_title("Acceleration")

    res_fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    main(parser.parse_args())