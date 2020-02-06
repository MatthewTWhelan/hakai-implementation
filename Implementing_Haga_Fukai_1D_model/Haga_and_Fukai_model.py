#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:42:05 2019

@author: Matt Whelan

The following is an attempt to replicate the results of Tatsuya Haga and Tomoki Fukai in their paper on reverse replay,
titled "Recurrent network model for learning goal-directed sequences through reverse replay".

This script contains two classes -- the first is the rate-coded linear network of Figure 1; the second is the rate
coded two dimensional network used for spatial navigation, Figures 7-10.
"""

import sys
import numpy as np
import pandas as pd


class LinearRecurrentNetwork(object):
    def __init__(self, weight_array=None, time_step=0.5, no_cells=500, w_max=27, d=5):
        '''

        :param weight_array: square numpy array, the weight matrix for the network
        :param time_step: float, the step size used for simulation in milliseconds
        :param w_max: float, sets the max value for weight initialisation
        :param d: float, sets the standard deviation for the weights
        '''

        self.time_step = time_step
        self.w_max = w_max
        self.d = d

        # Constants
        self.U = 0.6 # constant used in STP facilitation
        self.w_inh = 1.0

        # Initial conditions
        if weight_array is None:
            self.no_cells = no_cells
            self.weights = self.initialise_weights(self.no_cells, self.w_max, self.d)
            self.weights_next = np.zeros((self.no_cells, self.no_cells))
        else:
            self.no_cells = weight_array.shape[0]
            self.weights = weight_array.copy()
            self.weights_next = np.zeros((self.no_cells, self.no_cells))

        self.delta_w = np.zeros((self.no_cells, self.no_cells))
        self.delta_w_next = np.zeros((self.no_cells, self.no_cells))

        self.rates = np.zeros(self.no_cells)
        self.rates_next = np.zeros(self.no_cells)

        self.I_exc = np.zeros(self.no_cells)
        self.I_exc_next = np.zeros(self.no_cells)

        self.I_inh = np.zeros(self.no_cells)
        self.I_inh_next = np.zeros(self.no_cells)

        self.I_ext = np.zeros(self.no_cells)

        self.STP_D = np.ones(self.no_cells)
        self.STP_D_next = np.ones(self.no_cells)

        self.STP_F = np.ones(self.no_cells) * self.U
        self.STP_F_next = np.ones(self.no_cells) * self.U

    def initialise_weights(self, no_cells, w_max, d):
        '''

        :param no_cells: int, the number of cells in the network
        :param w_max: float, the max value for the weight
        :param d: float, the standard deviation used for setting the weights
        :return: numpy array of the weight matrix, where weights[i][j] specifies the weight from neuron i to neuron j
        '''

        weight_matrix = np.zeros((no_cells, no_cells))
        for j in range(no_cells):
            for i in range(no_cells):
                if i == j:
                    weight_matrix[i][j] = 0.0
                else:
                    weight_matrix[i][j] = w_max * np.exp(-np.abs(i - j) / d)
        return weight_matrix

    def update_cell_firing_rates(self, rho=0.0025, epsilon=0.5):
        '''

        :param I_ext: numpy array, the external current input
        :param rho: float, constant used for the linear rectifier unit
        :param epsilon: float, threshold in linear rectifier unit
        :return: None
        '''

        for i in range(self.no_cells):
            I = self.I_exc_next[i] - self.I_inh_next[i] + self.I_ext[i]
            if I > epsilon:
                self.rates_next[i] = rho * (I - epsilon)
            else:
                self.rates_next[i] = 0.0

    def update_STP(self, rates, STP_F, STP_D, tau_f=200.0, tau_d=500.0):
        '''

        :param rates: 1D numpy array of firing rates for the neurons
        :param STP_F: 1D numpy array of the Short Term Facilitation levels for the neurons
        :param STP_D: 1D numpy array of the Short Term Depression levels for the neurons
        :param tau_f: Float, STF time constant
        :param tau_d: Float, STD time constant
        :return: None
        '''

        for f in range(self.no_cells):
            self.STP_F_next[f] = ((self.U - STP_F[f]) / tau_f + self.U * (1 - STP_F[f]) * rates[f]) * self.time_step + \
                                 STP_F[f]
            self.STP_D_next[f] = ((1.0 - STP_D[f]) / tau_d - rates[f] * STP_D[f] * STP_F[f]) * self.time_step + STP_D[f]

    def update_currents(self, I_exc, I_inh, rates, weights, STP_F, STP_D, tau_exc = 10.0, tau_inh = 10.0):
        '''

        :param I_exc: 1D numpy array of excitation currents for the neurons
        :param I_inh: float, the global inhibition current
        :param rates: 1D numpy array of firing rates for the neurons
        :param weights: numpy array of the weight matrix
        :param STP_F: 1D numpy array of the Short Term Facilitation levels for the neurons
        :param STP_D: 1D numpy array of the Short Term Depression levels for the neurons
        :param tau_exc: float, excitatory time constant
        :param tau_inh: float, inhibitory time constant
        :return: None
        '''

        for f in range(self.no_cells):
            neuron_weight_inputs = weights[f]
            sum_exc = 0.0
            sum_inh = 0.0
            for j in range(self.no_cells):
                sum_exc += neuron_weight_inputs[j] * rates[j] * STP_D[j] * STP_F[j]
                sum_inh += self.w_inh * rates[j] * STP_D[j] * STP_F[j]
            self.I_exc_next[f] = (- I_exc[f] / tau_exc + sum_exc) * self.time_step + I_exc[f]
            self.I_inh_next[f] = (- I_inh[f] / tau_inh + sum_inh) * self.time_step + I_inh[f]

    def update_delta_w(self, delta_w, rates, STP_F, STP_D):
        '''

        :param delta_w: numpy array of the delta matrix
        :param rates: 1D numpy array of firing rates for the neurons
        :param STP_F: 1D numpy array of the Short Term Facilitation levels for the neurons
        :param STP_D: 1D numpy array of the Short Term Depression levels for the neurons
        :return: None
        '''

        eta = 20.0
        tau_w = 1000.0
        for j in range(self.no_cells):
            for i in range(self.no_cells):
                if i == j:
                    continue
                self.delta_w_next[i][j] = self.time_step * (- delta_w[i][j] + eta * rates[j] * rates[i] * STP_D[j] * STP_F[j]) \
                                     / tau_w + delta_w[i][j]

    def update_weights(self, delta_w, weights):
        '''

        :param delta_w: numpy array of the delta matrix
        :param weights: numpy array of the weight matrix
        :return: None
        '''

        for j in range(self.no_cells):
            for i in range(self.no_cells):
                if i == j:
                    continue
                self.weights_next[i][j] = delta_w[i][j] * self.time_step + weights[i][j]

    def begin_simulation(self, time_sim, ext_input=None):
        '''

        :param time_sim: float, the amount of time, in ms, the simulation should run for
        :param ext_input: function, the external input to the network as a function of time
        :returns: Pandas data frame of the network's rates (index is the neuron ID, column is the time),
        Pandas data frame of the weight matrix at the end of the simulation (both index and column represents neuron ID)
        '''

        network_firing_rates = {}
        no_steps = int(time_sim / self.time_step)
        for step in range(no_steps):
            if step != 0:
                self.weights = self.weights_next.copy()
                self.delta_w = self.delta_w_next.copy()
                self.rates = self.rates_next.copy()
                self.I_exc = self.I_exc_next.copy()
                self.I_inh = self.I_inh_next.copy()
                self.STP_D = self.STP_D_next.copy()
                self.STP_F = self.STP_F_next.copy()

            # call the external input function
            self.I_ext = np.zeros(self.no_cells)
            if ext_input is not None:
                ext_input(step)

            # update all the network variables
            self.update_cell_firing_rates()
            self.update_currents(self.I_exc, self.I_inh, self.rates, self.weights, self.STP_D, self.STP_F)
            self.update_STP(self.rates, self.STP_F, self.STP_D)
            self.update_delta_w(self.delta_w, self.rates, self.STP_D, self.STP_F)
            self.update_weights(self.delta_w, self.weights)

            network_firing_rates[step * self.time_step] = self.rates * 1000 # multiplied by 1000 for rate in units
            # per second

            print(int(step / no_steps * 100), "%", end="\r")

        # Creating the Pandas data frames for the rates and weights
        rates_df = pd.DataFrame.from_dict(network_firing_rates)
        weights_df = pd.DataFrame(self.weights, np.arange(self.no_cells), np.arange(self.no_cells))

        return rates_df, weights_df

    def ext_input_full(self, step):
        # excite the first 10 cells for the first 10ms
        if step * self.time_step < 10:
            self.I_ext[0:5] = 5

        # excite the middle 10 cells at 3s
        if 3000 < step * self.time_step < 3010:
            self.I_exc[244:254] = 5

    def ext_input_start(self, step):
        '''
        Stimulates the first 10 neurons for 10ms
        '''

        if step * self.time_step < 10:
            self.I_ext[0:5] = 5


if __name__ == "__main__":
    # Path to store data files
    data_dir = 'data/'

    sim_id = 5
    if sim_id == 1:
        # Running the standard 4s simulation
        print('Running the standard 4s simulation')
        network_standard = LinearRecurrentNetwork(time_step=0.5)
        time_sim = 4000 # ms
        rates, weights = network_standard.begin_simulation(time_sim, network_standard.ext_input_full)
        rates.to_csv(data_dir + 'standard_sim_neuron_rates_' + str(time_sim) + '.csv')
        weights.to_csv(data_dir + 'standard_sim_neuron_weights_' + str(time_sim) + '.csv')

    elif sim_id == 2:
        # Running the first 3s of the standard simulation
        print('Running the first 3s of the standard simulation')
        network_standard_three_sec = LinearRecurrentNetwork(time_step=0.5)
        time_sim = 3000  # ms
        rates, weights = network_standard_three_sec.begin_simulation(time_sim,
                                                                     network_standard_three_sec.ext_input_start)
        rates.to_csv(data_dir + 'standard_sim_neuron_rates_' + str(time_sim) + '.csv')
        weights.to_csv(data_dir + 'standard_sim_neuron_weights_' + str(time_sim) + '.csv')

    elif sim_id == 3:
        # Running an additional 1s of simulation following the standard 4s simulation
        print('Running an additional 1s of simulation following the standard 4s simulation')
        weights_initial = pd.DataFrame.from_csv(data_dir + 'standard_sim_neuron_weights_4000.csv').values
        network_additional_one_sec = LinearRecurrentNetwork(weight_array=weights_initial, time_step=0.5)
        time_sim = 1000
        rates, weights = network_additional_one_sec.begin_simulation(time_sim,
                                                                     network_additional_one_sec.ext_input_start)
        rates.to_csv(data_dir + 'additional_one_sec_neuron_rates.csv')
        weights.to_csv(data_dir + 'additional_one_sec_neuron_weights.csv')

    elif sim_id == 4:
        # Running an additional 4s of the standard simulation following the standard 4s simulation
        print('Running an additional 4s of standard simulation following the standard 4s simulation')
        weights_initial = pd.DataFrame.from_csv(data_dir + 'standard_sim_neuron_weights_4000.csv').values
        network_additional_four_sec = LinearRecurrentNetwork(weight_array=weights_initial, time_step=0.5)
        time_sim = 4000
        rates, weights = network_additional_four_sec.begin_simulation(time_sim,
                                                                      network_additional_four_sec.ext_input_full)
        rates.to_csv(data_dir + 'additional_four_sec_neuron_rates.csv')
        weights.to_csv(data_dir + 'additional_four_sec_neuron_weights.csv')

    elif sim_id == 5:
        # Running an additional 4s of the standard simulation following the 8s simulation
        print('Running an additional 4s of standard simulation following the 8s simulation')
        weights_initial = pd.DataFrame.from_csv(data_dir + 'additional_four_sec_neuron_weights.csv').values
        network_additional_four_sec = LinearRecurrentNetwork(weight_array=weights_initial, time_step=0.5)
        time_sim = 4000
        rates, weights = network_additional_four_sec.begin_simulation(time_sim,
                                                                      network_additional_four_sec.ext_input_full)
        rates.to_csv(data_dir + 'additional_eight_sec_neuron_rates.csv')
        weights.to_csv(data_dir + 'additional_eight_sec_neuron_weights.csv')
