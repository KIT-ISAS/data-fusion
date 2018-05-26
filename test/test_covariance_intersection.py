from unittest import TestCase, skip
from algorithms.covariance_intersection import CovarianceIntersection
import numpy as np
from plotting import plotting
import matplotlib.pyplot as plt


class TestCovarianceIntersection(TestCase):

    #@skip("Shows a plot - run manually")
    def test_covariance_intersection(self):
        # Create two random measurements
        seed = np.random.randint(0, 1000)
        #seed = 362
        print("Seed: {}".format(seed))
        np.random.seed(seed)
        random_matrix_a = np.random.rand(2,2)
        random_matrix_b = np.random.rand(2,2)
        measurement_cov_a = np.dot(random_matrix_a, random_matrix_a.T)
        measurement_cov_b = np.dot(random_matrix_b, random_matrix_b.T)
        estimate_a = np.random.random_sample()
        estimate_b = np.random.random_sample()

        ci = CovarianceIntersection()
        fused_mean, fused_cov = ci.fuse(estimate_a, measurement_cov_a, estimate_b, measurement_cov_b)
        fig = plt.figure(0)
        axes = fig.add_subplot(111)
        for index, cov in enumerate([measurement_cov_a, measurement_cov_b, fused_cov]):
            plotting.plot_covariance_ellipse(axes, cov, "C{}".format(index))
        axes.autoscale()
        axes.set_aspect("auto")
        axes.legend(["A", "B", "CI"])
        fig.show()