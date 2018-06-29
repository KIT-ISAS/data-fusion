from unittest import TestCase, skip
from simulation.process_models.constant_acceleration import ConstantAcceleration
import numpy as np
import random
import matplotlib.pyplot as plt


class TestConstantAcceleration(TestCase):
    def test_simple_model(self):
        for j in range(10):
            acc = random.uniform(-100, 100)
            vel = random.uniform(-100, 100)
            pos = random.uniform(-100, 100)
            initial_state = np.array([pos, vel, acc])
            model = ConstantAcceleration(initial_state.reshape((3, 1)), delta_t=1, noise_variance=0)
            model.step()
            for i in range(1, 10):
                current_state = np.squeeze(np.asarray(model.current_state))
                self.assertAlmostEqual(current_state[2], acc)
                self.assertAlmostEqual(current_state[1], vel + acc * i)
                self.assertAlmostEqual(current_state[0], pos + vel * i + 0.5 * acc * (i ** 2))
                model.step()

    def test_plot(self):
        acc = random.uniform(-10, 10)
        vel = random.uniform(-10, 10)
        pos = random.uniform(-10, 10)
        initial_state = np.array([pos, vel, acc])
        model = ConstantAcceleration(initial_state.reshape((3, 1)), delta_t=0.01, noise_variance=1000)
        for i in range(100):
            model.step()
        fig = plt.figure()
        axes = fig.add_subplot(111)
        model.plot(axes)
        plt.show()
