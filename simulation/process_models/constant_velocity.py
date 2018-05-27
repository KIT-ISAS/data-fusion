"""
Constant-velocity process model (2D state space)
"""

import numpy as np
import matplotlib.pyplot as plt


class ConstantVelocity(object):
    def __init__(self, initial_state, delta_t=0.1, noise_variance=10):
        self.F = np.matrix([[1.0, delta_t],
                            [0, 1.0]])
        self.G = np.array([[(delta_t ** 2) / 2.0],
                           [delta_t]])
        self.noise_stddev = np.sqrt(noise_variance)
        self.current_state = initial_state
        self.states = []

    def step(self):
        """
        Updates the current state.
        """
        process_noise = np.random.normal(0, scale=self.noise_stddev)
        self.current_state = np.dot(self.F, self.current_state) + np.multiply(self.G, process_noise)
        self.states.append(self.current_state)

    def plot(self, axes):
        """
        Creates a pyplot figure of the model's state history.
        """
        states = list(map(lambda x: np.squeeze(np.asarray(x)), self.states))
        positions = [state[0] for state in states]
        velocities = [state[1] for state in states]
        axes.plot(positions, label="Position")
        axes.plot(velocities, label="Velocity")
        axes.legend()
        return axes
