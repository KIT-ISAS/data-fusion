from unittest import TestCase, skip
from algorithms.covariance_intersection import CovarianceIntersection
from algorithms.naive import Naive
from algorithms.ellipsoidal_intersection import EllipsoidalIntersection
from algorithms.inverse_covariance_intersection import InverseCovarianceIntersection
import numpy as np
from plotting import plotting
import matplotlib.pyplot as plt


class TestFusionAlgorithms(TestCase):

    def test_fusion_algorithms(self):
        # Create two random measurements
        seed = np.random.randint(0, 1000)
        seed = 810
        print("Seed: {}".format(seed))
        np.random.seed(seed)
        random_matrix_a = np.random.rand(2,2)
        random_matrix_b = np.random.rand(2,2)
        measurement_cov_a = np.dot(random_matrix_a, random_matrix_a.T)
        measurement_cov_b = np.dot(random_matrix_b, random_matrix_b.T)
        estimate_a = np.random.rand(2,1)
        estimate_b = np.random.rand(2,1)

        fusion_algorithms = [Naive(), CovarianceIntersection(), EllipsoidalIntersection(), InverseCovarianceIntersection()]
        fig = plt.figure(0)
        axes = fig.add_subplot(111)
        for index, fusion_alg in enumerate(fusion_algorithms):
            fused_mean, fused_cov = fusion_alg.fuse(estimate_a, measurement_cov_a, estimate_b, measurement_cov_b)
            plotting.plot_covariance_ellipse(axes, fused_cov, "C{}".format(index))
        for index, cov in enumerate([measurement_cov_a, measurement_cov_b]):
            plotting.plot_covariance_ellipse(axes, cov, "C{}".format(len(fusion_algorithms) + index), linestyle="dashed", fill_alpha=0.2)
        axes.autoscale()
        axes.set_aspect("auto")
        labels = [alg.algorithm_abbreviation for alg in fusion_algorithms]
        labels.extend(["A", "B"])
        axes.legend(labels)
        fig.show()