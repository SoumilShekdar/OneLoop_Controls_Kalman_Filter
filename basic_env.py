# Environment Setup
# Objective: Simulate Motion of the Hyperloop Pod
# Key Components: 1) Equation to Simulate true motion of the Hyperloop Pod
#                 2) Equation to model measured Position
#                 3) Physics Based Expected Position and Velocity Equation
# Assumptions: - Sample Measured position with 1m standard deviation
#              - Sample Measured velocity with 0.1km/hr standard deviaation
#              - Constant Accelaration

import argparse
import numpy as np
import random 
import matplotlib.pyplot as plt 
from scipy import optimize

class Sampler:
    def __init__(self, standard_deviation = 0):
        self.mean = 0
        self.standard_deviation = standard_deviation
        self.sample = []

    def get_sample_value(self, true_value):
        self.mean = true_value
        self.sample = np.random.normal(self.mean, self.standard_deviation, 100)
        return self.sample[np.random.randint(100)]

class KalmanFilter:
    def __init__(self, Q = 0.1, R = 0.1, H = 1, A = 1):
        self.q = Q
        self.r = R
        self.h = H
        self.a = A
        self.number_of_steps = 0
        self.p = 0
    
    def setup(self, no_steps, p_value = 1):
        self.number_of_steps = no_steps
        self.p = p_value

    def update_params(self, params):
        self.q = params[0]
        self.r = params[1]
        self.h = params[2]
        self.a = params[3]

    def get_params(self):
        return [self.q, self.r, self.h, self.a]

    def update(self, calculated_value, measured_value):
        temp_p = self.a * self.a * self.p + self.q
        kalman_gain = self.h * temp_p / (self.h * self.h * temp_p + self.r)
        kalman_value = calculated_value + kalman_gain * (measured_value - self.h * calculated_value)
        self.p = (1 - kalman_gain * self.h ) * temp_p
        return kalman_value

class Simulation:
    def __init__(self, total_time, time_step):
        self.total_time = total_time
        self.time_step = time_step
        self.no_steps = int(total_time / time_step + 1)
        self.time = np.linspace(0, self.total_time, self.no_steps)

        self.velocity = np.zeros(self.no_steps)
        self.true_position = np.zeros(self.no_steps)
        self.measured_position = np.zeros(self.no_steps)
        self.calculate_position = np.zeros(self.no_steps)
        self.expected_velocity = np.zeros(self.no_steps)
        self.kalman_position = np.zeros(self.no_steps)

        self.velocity_measure = None
        self.position_meausre = None


        self.error_kalman_true = np.zeros(self.no_steps)
        self.error_calculate_true = np.zeros(self.no_steps)
        self.error_measure_true = np.zeros(self.no_steps)

        self.kalman_filter = KalmanFilter()

    def setup(self, expected_velocity, start_time, velocity_deviation, position_deviation):
        self.expected_velocity[int(start_time):] = expected_velocity
        self.velocity_measure = Sampler(velocity_deviation)
        self.position_measure = Sampler(position_deviation)
        self.kalman_filter.setup(self.no_steps)

    def sum(self, list):
        total = 0
        for item in list:
            total = total + item
        return total

    def fit(self, initial_values):
        kalman_filter = KalmanFilter(initial_values[0], initial_values[1], initial_values[2], initial_values[3])
        kalman_filter.setup(self.no_steps)
        true_position = np.zeros(self.no_steps)
        measured_position = np.zeros(self.no_steps)
        velocity = np.zeros(self.no_steps)
        calculate_position = np.zeros(self.no_steps)
        kalman_position = np.zeros(self.no_steps)
        error = np.zeros(self.no_steps)
        for i in range(self.no_steps - 1):
            true_position[i] = true_position[i - 1] + self.expected_velocity[i - 1] * self.time_step
            measured_position[i] = self.position_measure.get_sample_value(true_position[i])
            velocity[i] = self.velocity_measure.get_sample_value(self.expected_velocity[i]) 
            calculate_position[i] = calculate_position[i - 1] + velocity[i] * self.time_step 
            kalman_position[i] = kalman_filter.update(calculate_position[i], measured_position[i])
            error[i] = abs(kalman_position[i] - true_position[i])
        return sum(error)
    
    def simulate(self):
        self.kalman_filter.update_params(optimize.fmin(self.fit, self.kalman_filter.get_params()))
        plt.figure(1, figsize = (5, 4))
        plt.ion()
        plt.show()

        for i in range(self.no_steps - 1):
            self.true_position[i] = self.true_position[i - 1] + self.expected_velocity[i - 1] * self.time_step
            self.measured_position[i] = self.position_measure.get_sample_value(self.true_position[i])
            self.velocity[i] = self.velocity_measure.get_sample_value(self.expected_velocity[i]) 
            self.calculate_position[i] = self.calculate_position[i - 1] + self.velocity[i] * self.time_step 
            self.kalman_position[i] = self.kalman_filter.update(self.calculate_position[i], self.measured_position[i])
            self.error_kalman_true[i] = abs(self.kalman_position[i] - self.true_position[i])
            self.error_calculate_true[i] = abs(self.calculate_position[i] - self.true_position[i])
            self.error_measure_true[i] = abs(self.measured_position[i] - self.true_position[i])

            plt.clf()
            
            plt.subplot(2, 1, 1)
            plt.plot(self.time[0: i + 1], self.measured_position[0: i + 1], 'b-', linewidth = 1)
            plt.plot(self.time[0: i + 1], self.calculate_position[0: i + 1], 'g-', linewidth = 1)
            plt.plot(self.time[0: i + 1], self.kalman_position[0: i + 1], 'm-', linewidth = 1)
            plt.plot(self.time[0: i + 1], self.true_position[0: i + 1], 'r-', linewidth = 1)
            plt.ylabel('Position')
            plt.legend(['Measured Position', 'Calculated Position', 'Kalman Position', 'True Position'], loc = 2)

            plt.subplot(2, 1, 2)
            plt.plot(self.time[0: i + 1], self.error_measure_true[0: i + 1], 'b-', linewidth = 1)
            plt.plot(self.time[0: i + 1], self.error_calculate_true[0: i + 1], 'g-', linewidth = 1)
            plt.plot(self.time[0: i + 1], self.error_kalman_true[0: i + 1], 'r-', linewidth = 1)
            plt.ylabel('Error')
            plt.legend(['Measured Error', 'Calculated Error', 'Kalman Error'], loc = 2)
            plt.xlabel('Time')

            plt.pause(0.1)

        plt.savefig("result.png")
        print(self.sum(self.error_measure_true))
        print(self.sum(self.error_calculate_true))
        print(self.sum(self.error_kalman_true))


parser = argparse.ArgumentParser()
parser.add_argument("--sensors", nargs = '+', type = float, help = "velocity_deviation position_deviation")
parser.add_argument("--sim", nargs = '+', type = float, help = "total_time start_time expected_velocity")

args = parser.parse_args()
sim = Simulation(args.sim[0], 1.0)
sim.setup(args.sim[2], args.sim[1], args.sensors[0], args.sensors[1])

sim.simulate()  