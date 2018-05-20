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
    timesteps = 50 if args.timesteps is None else args.timesteps
    print("Timesteps: {}".format(timesteps))
    fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
    process_states = []
    local_estimates = {}
    fused_estimates = {}
    for i, fusion_algorithm in enumerate(fusion_algorithms):
        np.random.seed(seed)
        sim = SimpleSensorNetworkSimulation(fusion_algorithm)
        print("Running simulation {} ({})".format(i, fusion_algorithm.algorithm_abbreviation))
        sim.run(timesteps)
        if i == 0:
            process_states = list(map(lambda x: np.squeeze(np.asarray(x)), sim.process.states))
            # Plot process
            plt.rcParams["figure.figsize"] = (4, 3)
            proc_fig = plt.figure(1)
            proc_axes = proc_fig.add_subplot(111)
            sim.process.plot(proc_axes)
            proc_fig.show()

            # Save local estimates
            local_estimates["A"] = sim.node_a.local_estimates
            local_estimates["B"] = sim.node_b.local_estimates
        fused_estimates[fusion_algorithm.algorithm_abbreviation] = {
            "A": sim.node_a.fused_estimates,
            "B": sim.node_b.fused_estimates
        }

    # Plot results
    pos_real = [x[0] for x in process_states]
    vel_real = [x[1] for x in process_states]
    acc_real = [x[2] for x in process_states]
    pos_fused = {}
    vel_fused = {}
    acc_fused = {}
    for alg in fusion_algorithms:
        pos_fused[alg.algorithm_abbreviation] = [x[0][0] for x in fused_estimates[alg.algorithm_abbreviation]["A"]]
        vel_fused[alg.algorithm_abbreviation] = [x[0][1] for x in fused_estimates[alg.algorithm_abbreviation]["A"]]
        acc_fused[alg.algorithm_abbreviation] = [x[0][2] for x in fused_estimates[alg.algorithm_abbreviation]["A"]]

    plt.rcParams["figure.figsize"] = (8, 8)
    res_fig = plt.figure(2)
    pos_axes = res_fig.add_subplot(3, 1, 1)
    pos_axes.plot(pos_real, label="Real")
    for alg in pos_fused.keys():
        pos_axes.plot(pos_fused[alg], label=alg)
    pos_axes.legend()
    pos_axes.set_title("Position")

    vel_axes = res_fig.add_subplot(3, 1, 2)
    vel_axes.plot(vel_real, label="Real")
    for alg in vel_fused.keys():
        vel_axes.plot(vel_fused[alg], label=alg)
    vel_axes.legend()
    vel_axes.set_title("Velocity")

    acc_axes = res_fig.add_subplot(3, 1, 3)
    acc_axes.plot(acc_real, label="Real")
    for alg in acc_fused.keys():
        acc_axes.plot(acc_fused[alg], label=alg)
    acc_axes.legend()
    acc_axes.set_title("Acceleration")

    res_fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, help="Number of timesteps")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    main(parser.parse_args())