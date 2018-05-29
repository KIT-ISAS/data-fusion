import argparse
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

from plotting import plotting
from simulation.sensors.kalman_sensor_node import KalmanSensorNode
from simulation.process_models.constant_process import ConstantProcess
from simulation.simple_sensor_network import SimpleSensorNetworkSimulation
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
from algorithms.naive import Naive

import matplotlib.pyplot as plt

class ConsistencyExperiment(object):
    def __init__(self, fusion_algorithms):
        self.fusion_algorithms = fusion_algorithms

    def run_trial(self, _):
        error_covs = {}
        reported_covs = {}
        seed = np.random.randint(0, 100000)
        for idx, alg in enumerate(self.fusion_algorithms):
            np.random.seed(seed)
            process = ConstantProcess(initial_state=[0, 0], noise_variance=2)
            measurement_covariance = np.multiply(np.identity(2), 0.2)
            node_a = KalmanSensorNode(0, 2, process, alg, np.identity(2), process.covariance, measurement_covariance)
            node_b = KalmanSensorNode(0, 2, process, alg, np.identity(2), process.covariance, measurement_covariance)
            sim = SimpleSensorNetworkSimulation(alg, process, (node_a, node_b))
            sim.run(1)
            process_state = process.current_state
            fused_mean, fused_cov = sim.node_a.fused_estimates[0]
            error_vec = fused_mean - process_state
            error_covs[alg.algorithm_abbreviation] = np.outer(error_vec, error_vec)
            reported_covs[alg.algorithm_abbreviation] = fused_cov
        return error_covs, reported_covs

    def run(self, runs):
        print("Running Monte Carlo experiment ({} runs)...".format(runs))

        pool = Pool()
        res = pool.map(self.run_trial, range(runs))
        pool.close()
        pool.join()
        error_covs, reported_covs = map(list, zip(*res))

        for alg in self.fusion_algorithms:
            abbr = alg.algorithm_abbreviation
            mean_error_cov = np.mean([error_covs[i][abbr] for i in range(runs)], 0)
            fig = plt.figure()
            axes = fig.add_subplot(111)
            plotting.plot_covariance_ellipse(axes, mean_error_cov, "C0", linestyle="dashed")
            plotting.plot_covariance_ellipse(axes, reported_covs[0][abbr], "C1", linestyle="solid")
            axes.autoscale()
            axes.set_aspect("auto")
            labels = ["Actual", abbr]
            axes.legend(labels)
            fig.show()



def main(args):
    fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
    experiment = ConsistencyExperiment(fusion_algorithms)
    experiment.run(args.runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int)
    main(parser.parse_args())