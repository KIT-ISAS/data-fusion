from unittest import TestCase, skip
from simulation.process_models.constant_velocity import ConstantVelocity
import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import rcParams


class TestConstantVelocity(TestCase):
    def test_simple_model(self):
        for j in range(10):
            pos = random.uniform(-100, 100)
            vel = random.uniform(-100, 100)
            initial_state = np.array([pos, vel])
            model = ConstantVelocity(initial_state.reshape((2, 1)), delta_t=1, noise_variance=0)
            model.step()
            for i in range(1, 10):
                current_state = np.squeeze(np.asarray(model.current_state))
                self.assertAlmostEqual(current_state[1], vel)
                self.assertAlmostEqual(current_state[0], pos + vel * i)
                model.step()

    def test_plot(self):
        #vel = random.uniform(-10, 10)
        #pos = random.uniform(-10, 10)
        pos = 0
        vel = 0
        initial_state = np.array([pos, vel])
        model = ConstantVelocity(initial_state.reshape((2, 1)), delta_t=0.01, noise_variance=1000)
        for i in range(100):
            model.step()
        rcParams["figure.figsize"] = 3.5,2.5
        fig = plt.figure()
        axes = fig.add_subplot(111)
        model.plot(axes)
        plt.xlabel("Timestep")
        plt.show()
