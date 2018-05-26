from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.naive import Naive

from plotting import plotting

import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from numpy.linalg import inv


def plot_results(fusion_algorithms, process_states, fused_estimates):
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


def plot_ellipses(fusion_algorithms, local_estimates, fused_estimates):
    # Plot covariance ellipses
    plt.rcParams["figure.figsize"] = (4, 4)
    ellipse_fig = plt.figure(3)
    ellipse_axes = ellipse_fig.add_subplot(1, 1, 1)
    for i, alg in enumerate(fused_estimates.keys()):
        plotting.plot_covariance_ellipse(ellipse_axes, fused_estimates[alg]["A"][-1][1], "C{}".format(i))
    plotting.plot_covariance_ellipse(ellipse_axes, local_estimates["A"][-1][1], "C{}".format(len(fusion_algorithms)), "dotted",
                            0.2)
    plotting.plot_covariance_ellipse(ellipse_axes, local_estimates["B"][-1][1], "C{}".format(len(fusion_algorithms) + 1),
                            "dotted", 0.2)
    ellipse_axes.autoscale()
    ellipse_axes.set_aspect("auto")
    labels = list(fused_estimates.keys())
    labels.extend(["A", "B"])
    ellipse_axes.legend(labels)
    ellipse_fig.show()
    ellipse_fig.clf()

    # Plot inverse covariance ellipses
    plt.rcParams["figure.figsize"] = (4, 4)
    inverse_ellipse_fig = plt.figure(4)
    inverse_ellipse_axes = inverse_ellipse_fig.add_subplot(1, 1, 1)
    for i, alg in enumerate(fused_estimates.keys()):
        plotting.plot_covariance_ellipse(inverse_ellipse_axes, inv(fused_estimates[alg]["A"][-1][1]), "C{}".format(i))
    plotting.plot_covariance_ellipse(inverse_ellipse_axes, inv(local_estimates["A"][-1][1]),
                            "C{}".format(len(fusion_algorithms)), "dotted", 0.2)
    plotting.plot_covariance_ellipse(inverse_ellipse_axes, inv(local_estimates["B"][-1][1]),
                            "C{}".format(len(fusion_algorithms) + 1), "dotted", 0.2)
    inverse_ellipse_axes.autoscale()
    inverse_ellipse_axes.set_aspect("auto")
    labels = list(fused_estimates.keys())
    labels.extend(["A", "B"])
    inverse_ellipse_axes.legend(labels)
    inverse_ellipse_fig.show()
    inverse_ellipse_fig.clf()


def plot_process(process):
    plt.rcParams["figure.figsize"] = (4, 3)
    proc_fig = plt.figure(1)
    proc_axes = proc_fig.add_subplot(111)
    process.plot(proc_axes)
    proc_fig.show()


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
            plot_process(sim.process)
            # Save local estimates
            local_estimates["A"] = sim.node_a.local_estimates
            local_estimates["B"] = sim.node_b.local_estimates
        # Save fused estimates
        fused_estimates[fusion_algorithm.algorithm_abbreviation] = {
            "A": sim.node_a.fused_estimates,
            "B": sim.node_b.fused_estimates
        }
    plot_results(fusion_algorithms, process_states, fused_estimates)
    plot_ellipses(fusion_algorithms, local_estimates, fused_estimates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example for how to use the package. Runs the SimpleSensorNetwork simulation to evaluate different fusion algorithms.")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    main(parser.parse_args())