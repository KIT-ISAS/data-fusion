import random
import numpy as np
import argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time

from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.covariance_intersection import PerformanceCriterion
from algorithms.naive import Naive


class MonteCarloExperiment(object):
    def __init__(self, fusion_algorithms, simulation_class):
        self.fusion_algorithms = fusion_algorithms
        self.simulation_class = simulation_class

    @staticmethod
    def is_consistent(estimate, process_state):
        error_vec = estimate[0] - process_state
        mat = estimate[1] - np.dot(error_vec, error_vec.T)
        pos_semidefinite = np.all(np.linalg.eigvals(mat) >= 0)
        return pos_semidefinite

    def plot_mean_squared_errors(self, pos_squared_errors, vel_squared_errors, node="A"):
        timesteps = len(pos_squared_errors[0]["Naive"])
        pos_mse = {}
        vel_mse = {}
        for alg in self.fusion_algorithms:
            pos_mse[alg.algorithm_abbreviation] = np.mean([pos_squared_errors[i][alg.algorithm_abbreviation] for i in range(len(pos_squared_errors))], axis=0)
            vel_mse[alg.algorithm_abbreviation] = np.mean([vel_squared_errors[i][alg.algorithm_abbreviation] for i in range(len(vel_squared_errors))], axis=0)

        for end_timestep in range(10, timesteps + 10, 10):
            plt.rcParams["figure.figsize"] = (6, 5)
            res_fig = plt.figure()
            pos_axes = res_fig.add_subplot(2, 1, 1)
            for alg in pos_mse.keys():
                pos_axes.plot(pos_mse[alg][:end_timestep], label=alg)
            pos_axes.legend()
            pos_axes.set_title("Position MSE (Node {})".format(node))
            pos_axes.set_xlabel("Timestep")

            vel_axes = res_fig.add_subplot(2, 1, 2)
            for alg in vel_mse.keys():
                vel_axes.plot(vel_mse[alg][:end_timestep], label=alg)
            vel_axes.legend()
            vel_axes.set_title("Velocity MSE (Node {})".format(node))
            vel_axes.set_xlabel("Timestep")

            res_fig.show()

        # Zoom view
        plt.rcParams["figure.figsize"] = (6, 5)
        res_fig = plt.figure()
        pos_axes = res_fig.add_subplot(2, 1, 1)
        max_y = 0
        min_y = 100000
        for idx, alg in enumerate(pos_mse.keys()):
            if not (alg == "EI" or alg == "Naive"):
                pos_axes.plot(pos_mse[alg], color="C{}".format(idx), label=alg)
                max_y = max(max_y, max(pos_mse[alg][80:timesteps]))
                min_y = min(min_y, min(pos_mse[alg][80:timesteps]))
        pos_axes.legend()
        pos_axes.set_title("Position MSE (Node {})".format(node))
        pos_axes.set_xlabel("Timestep")
        pos_axes.set_xlim(80, timesteps - 1)
        pos_axes.set_ylim(min_y, max_y)

        vel_axes = res_fig.add_subplot(2, 1, 2)
        max_y = 0
        min_y = 100000
        for idx, alg in enumerate(vel_mse.keys()):
            if not (alg == "EI" or alg == "Naive"):
                vel_axes.plot(vel_mse[alg], color="C{}".format(idx), label=alg)
                max_y = max(max_y, max(vel_mse[alg][80:timesteps]))
                min_y = min(min_y, min(vel_mse[alg][80:timesteps]))
        vel_axes.legend()
        vel_axes.set_title("Velocity MSE (Node {})".format(node))
        vel_axes.set_xlabel("Timestep")
        vel_axes.set_xlim(80, timesteps - 1)
        vel_axes.set_ylim(min_y, max_y)

        res_fig.show()

    def run_trial(self, timesteps):
        seed = random.randint(0, 10000)
        process_states = []
        pos_squared_errors = {}
        vel_squared_errors = {}

        for i, fusion_algorithm in enumerate(self.fusion_algorithms):
            np.random.seed(seed)
            sim = self.simulation_class(fusion_algorithm)
            sim.run(timesteps)
            if i == 0:
                process_states = list(map(lambda x: np.squeeze(np.asarray(x)), sim.process.states))
            # Save fused estimates
            fused_estimates = sim.node_a.fused_estimates
            pos_squared_errors[fusion_algorithm.algorithm_abbreviation] = [(fused_estimates[i][0][0] - process_states[i][0]) ** 2 for i in range(timesteps)]
            vel_squared_errors[fusion_algorithm.algorithm_abbreviation] = [(fused_estimates[i][0][1] - process_states[i][1]) ** 2 for i in range(timesteps)]

        return pos_squared_errors, vel_squared_errors

    def run(self, runs, timesteps):
        print("Running Monte Carlo simulation with {} runs à {} timesteps...".format(runs, timesteps))
        pool = Pool()
        res = pool.map(self.run_trial, [timesteps for i in range(runs)])
        pool.close()
        pool.join()
        pos_squared_errors, vel_squared_errors = map(list, zip(*res))
        self.plot_mean_squared_errors(pos_squared_errors, vel_squared_errors)


def main(args):
    start_time = time.time()
    fusion_algorithms = [Naive(), CovarianceIntersection(PerformanceCriterion.DETERMINANT), EllipsoidalIntersection(), InverseCovarianceIntersection(PerformanceCriterion.DETERMINANT)]
    experiment = MonteCarloExperiment(fusion_algorithms, SimpleSensorNetworkSimulation)
    experiment.run(args.runs, args.timesteps)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int)
    parser.add_argument("timesteps", type=int)
    main(parser.parse_args())
