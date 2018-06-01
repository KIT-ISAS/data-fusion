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


def plot_results(fusion_algorithms, process_states, fused_estimates, node="A"):
    pos_real = [x[0] for x in process_states]
    vel_real = [x[1] for x in process_states]
    pos_fused = {}
    vel_fused = {}
    for alg in fusion_algorithms:
        pos_fused[alg.algorithm_abbreviation] = [x[0][0] for x in fused_estimates[alg.algorithm_abbreviation][node]]
        vel_fused[alg.algorithm_abbreviation] = [x[0][1] for x in fused_estimates[alg.algorithm_abbreviation][node]]

    plt.rcParams["figure.figsize"] = (6, 4)
    res_fig = plt.figure(2)
    pos_axes = res_fig.add_subplot(2, 1, 1)
    pos_axes.plot(pos_real, color="black", linestyle="dashed", label="Real")
    for alg in pos_fused.keys():
        pos_axes.plot(pos_fused[alg], label=alg)
    pos_axes.legend()
    pos_axes.set_title("Position (Node {})".format(node))
    pos_axes.set_xlabel("Timestep")

    vel_axes = res_fig.add_subplot(2, 1, 2)
    vel_axes.plot(vel_real, color="black", linestyle="dashed", label="Real")
    for alg in vel_fused.keys():
        vel_axes.plot(vel_fused[alg], label=alg)
    vel_axes.legend()
    vel_axes.set_title("Velocity (Node {})".format(node))
    vel_axes.set_xlabel("Timestep")

    res_fig.show()


def plot_estimation_errors(fusion_algorithms, process_states, fused_estimates, node="A"):
    pos_real = [x[0] for x in process_states]
    vel_real = [x[1] for x in process_states]
    pos_fused = {}
    vel_fused = {}
    for alg in fusion_algorithms:
        pos_fused[alg.algorithm_abbreviation] = [x[0][0] for x in fused_estimates[alg.algorithm_abbreviation][node]]
        vel_fused[alg.algorithm_abbreviation] = [x[0][1] for x in fused_estimates[alg.algorithm_abbreviation][node]]
    pos_squared_errors = {}
    vel_squared_errors = {}
    for alg in fusion_algorithms:
        alg_abbr = alg.algorithm_abbreviation
        pos_squared_errors[alg.algorithm_abbreviation] = [(pos_fused[alg_abbr][idx] - pos_real[idx])**2 for idx in range(len(pos_real))]
        vel_squared_errors[alg.algorithm_abbreviation] = [(vel_fused[alg_abbr][idx] - vel_real[idx])**2 for idx in range(len(vel_real))]

    #plt.rcParams["figure.figsize"] = (8, 8)
    res_fig = plt.figure()
    pos_axes = res_fig.add_subplot(2, 1, 1)
    for alg in pos_squared_errors.keys():
        pos_axes.plot(pos_squared_errors[alg], label=alg)
    pos_axes.legend()
    pos_axes.set_title("Squared Error (Position) (Node {})".format(node))
    pos_axes.set_xlabel("Timestep")

    vel_axes = res_fig.add_subplot(2, 1, 2)
    for alg in vel_squared_errors.keys():
        vel_axes.plot(vel_squared_errors[alg], label=alg)
    vel_axes.legend()
    vel_axes.set_title("Squared Error (Velocity) (Node {})".format(node))
    vel_axes.set_xlabel("Timestep")

    res_fig.show()


def plot_ellipses(fusion_algorithms, local_estimates, fused_estimates):
    # Plot covariance ellipses
    plt.rcParams["figure.figsize"] = (4, 4)
    ellipse_fig = plt.figure()
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
    """
    plt.rcParams["figure.figsize"] = (4, 4)
    inverse_ellipse_fig = plt.figure()
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
    """


def plot_process(process):
    plt.rcParams["figure.figsize"] = (4, 3)
    proc_fig = plt.figure()
    proc_axes = proc_fig.add_subplot(111)
    proc_axes.set_xlabel("Timestep")
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
    plot_estimation_errors(fusion_algorithms, process_states, fused_estimates)
    #plot_ellipses(fusion_algorithms, local_estimates, fused_estimates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example for how to use the package. Runs the SimpleSensorNetwork simulation to evaluate different fusion algorithms.")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    main(parser.parse_args())