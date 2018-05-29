"""
Constant process model (2D state space)
"""

import numpy as np


class ConstantProcess(object):
    def __init__(self, initial_state, noise_variance=10):
        self.current_state = initial_state
        self.covariance = np.multiply(np.identity(2), noise_variance)
        self.states = []

    def step(self):
        """
        Updates the current state.
        """
        self.current_state = self.current_state + np.random.multivariate_normal([0,0], self.covariance)
        self.states.append(self.current_state)
