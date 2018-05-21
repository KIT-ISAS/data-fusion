import random
import numpy as np
import argparse
from multiprocessing import Pool

from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
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

    def run_trial(self, timesteps):
        seed = random.randint(0, 10000)
        process_states = []
        local_estimates = {}
        fused_estimates = {}
        consistent = {}

        for i, fusion_algorithm in enumerate(self.fusion_algorithms):
            np.random.seed(seed)
            sim = self.simulation_class(fusion_algorithm)
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
            consistent[fusion_algorithm.algorithm_abbreviation] = [self.is_consistent(fused_estimates[fusion_algorithm.algorithm_abbreviation]["A"][ts], process_states[ts]) for ts in
                               range(timesteps)]
        return consistent

    def run(self, runs, timesteps):
        self.trials_completed = 0
        pool = Pool()
        consistent = pool.map(self.run_trial, [timesteps for i in range(runs)])
        pool.close()
        pool.join()

        for alg in self.fusion_algorithms:
            abbr = alg.algorithm_abbreviation
            consistent_count = 0
            for run in range(runs):
                run_res = consistent[run][abbr]
                for timestep in range(timesteps):
                    if run_res[timestep]:
                        consistent_count += 1
            percentage_consistent = (consistent_count / (runs * timesteps)) * 100
            print("{}: {}% ({} / {})".format(abbr, percentage_consistent, consistent_count, runs * timesteps))


def main(args):
    fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
    experiment = MonteCarloExperiment(fusion_algorithms, SimpleSensorNetworkSimulation)
    experiment.run(args.runs, args.timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int)
    parser.add_argument("timesteps", type=int)
    main(parser.parse_args())
