"""
Constant-acceleration process model according to Julier & Uhlmann (http://discovery.ucl.ac.uk/135541/)
"""

import numpy as np


class ProcessModel(object):
    def __init__(self, initial_state, delta_t=0.1, noise_variance=10):
        self.F = np.matrix([[1.0, delta_t, 0.5 * (delta_t ** 2)],
                            [0, 1.0, delta_t],
                            [0, 0, 1]])
        self.G = np.array([(delta_t ** 3) / 6.0,
                           (delta_t ** 2) / 2.0,
                           delta_t])
        self.noise_stddev = np.sqrt(noise_variance)
        self.current_state = initial_state

    def step(self):
        self.current_state = self.F * self.current_state + self.G * np.random.normal(0, scale=self.noise_stddev)
        return self.current_state