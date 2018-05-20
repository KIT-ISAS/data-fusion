from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.naive import Naive

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import random
import argparse
import math


def ellipse_bounding_box(w, h, theta):
    ux = w * math.cos(theta)
    uy = w * math.sin(theta)
    vx = h * math.cos(theta + math.pi / 2.0)
    vy = h * math.sin(theta + math.pi / 2.0)
    bbox_halfwidth = math.sqrt(ux * ux + vx * vx)
    bbox_halfheight = math.sqrt(uy * uy + vy * vy)
    return bbox_halfwidth, bbox_halfheight


def plot_covariance_ellipse(axes, cov, color, linestyle="solid"):
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(cov)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:2, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals[:2])
    axes.tick_params(axis='both', which='major')#, labelsize=20)
    ellipse = Ellipse([0,0], w, h, theta, linestyle=linestyle, linewidth=1.5, color=color, fill=False)
    ellipse.set_clip_box(axes.bbox)
    axes.add_patch(ellipse)

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

    # Plot covariance ellipses
    plt.rcParams["figure.figsize"] = (4, 3)
    ellipse_fig = plt.figure(3)
    ellipse_axes = ellipse_fig.add_subplot(1, 1, 1)
    for i, alg in enumerate(fused_estimates.keys()):
        plot_covariance_ellipse(ellipse_axes, fused_estimates[alg]["A"][-1][1], "C{}".format(i))
    plot_covariance_ellipse(ellipse_axes, local_estimates["A"][-1][1], "C{}".format(len(fusion_algorithms)), "dashed")
    plot_covariance_ellipse(ellipse_axes, local_estimates["B"][-1][1], "C{}".format(len(fusion_algorithms) + 1), "dashed")
    ellipse_axes.autoscale()
    ellipse_axes.set_aspect("auto")
    labels = list(fused_estimates.keys())
    labels.extend(["A","B"])
    ellipse_axes.legend(labels)
    ellipse_fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, help="Number of timesteps")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    main(parser.parse_args())