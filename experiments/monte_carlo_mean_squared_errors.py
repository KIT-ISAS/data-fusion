import random
import numpy as np
import argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt

from simulation.process_models.constant_velocity import ConstantVelocity
from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.naive import Naive


class MonteCarloMeanSquaredErrors(object):
    def __init__(self, fusion_algorithms, simulation_class, process):
        self.fusion_algorithms = fusion_algorithms
        self.simulation_class = simulation_class
        self.process = process

    @staticmethod
    def is_consistent(estimate, process_state):
        error_vec = estimate[0] - process_state
        mat = estimate[1] - np.dot(error_vec, error_vec.T)
        pos_semidefinite = np.all(np.linalg.eigvals(mat) >= 0)
        return pos_semidefinite

    def plot_mean_squared_errors(self, process_states, fused_estimates, node="A"):
        timesteps = len(process_states[0])
        pos_mse = {}
        vel_mse = {}
        for alg in self.fusion_algorithms:
            pos_squared_errors = []
            vel_squared_errors = []
            for run_idx, item in enumerate(fused_estimates):
               pos_fused = [item[alg.algorithm_abbreviation][node][i][0][0] for i in range(timesteps)]
               vel_fused = [item[alg.algorithm_abbreviation][node][i][0][1] for i in range(timesteps)]
               pos_squared_errors.append([(pos_fused[i] - process_states[run_idx][i][0])**2 for i in range(timesteps)])
               vel_squared_errors.append([(vel_fused[i] - process_states[run_idx][i][1])**2 for i in range(timesteps)])
            pos_mse[alg.algorithm_abbreviation] = np.mean(pos_squared_errors, axis=0)
            vel_mse[alg.algorithm_abbreviation] = np.mean(vel_squared_errors, axis=0)

        plt.rcParams["figure.figsize"] = (8, 8)
        res_fig = plt.figure()
        pos_axes = res_fig.add_subplot(2, 1, 1)
        for alg in pos_mse.keys():
            pos_axes.plot(pos_mse[alg], label=alg)
        pos_axes.legend()
        pos_axes.set_title("Position MSE (Node {})".format(node))

        vel_axes = res_fig.add_subplot(2, 1, 2)
        for alg in vel_mse.keys():
            vel_axes.plot(vel_mse[alg], label=alg)
        vel_axes.legend()
        vel_axes.set_title("Velocity MSE (Node {})".format(node))

        res_fig.show()


    def run_trial(self, timesteps):
        seed = random.randint(0, 10000)
        process_states = []
        local_estimates = {}
        fused_estimates = {}

        for i, fusion_algorithm in enumerate(self.fusion_algorithms):
            np.random.seed(seed)
            sim = self.simulation_class(fusion_algorithm, self.process)
            sim.run(timesteps)
            if i == 0:
                process_states = list(map(lambda x: np.squeeze(np.asarray(x)), sim.process.states))
                # Save local estimates
                local_estimates["A"] = sim.node_a.local_estimates
                local_estimates["B"] = sim.node_b.local_estimates
            # Save fused estimates
            fused_estimates[fusion_algorithm.algorithm_abbreviation] = {
                "A": sim.node_a.fused_estimates,
                "B": sim.node_b.fused_estimates
            }
        return process_states, fused_estimates

    def run(self, runs, timesteps):
        print("Running Monte Carlo simulation with {} runs à {} timesteps...".format(runs, timesteps))
        self.trials_completed = 0
        pool = Pool()
        res = pool.map(self.run_trial, [timesteps for i in range(runs)])
        pool.close()
        pool.join()
        process_states, fused_estimates = map(list, zip(*res))
        self.plot_mean_squared_errors(process_states, fused_estimates)


def main(args):
    fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
    process = ConstantVelocity(initial_state=np.zeros(shape=(2, 1)), delta_t=0.01)
    experiment = MonteCarloMeanSquaredErrors(fusion_algorithms, SimpleSensorNetworkSimulation, process)
    experiment.run(args.runs, args.timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int)
    parser.add_argument("timesteps", type=int)
    main(parser.parse_args())
