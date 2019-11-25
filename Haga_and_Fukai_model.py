#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:42:05 2019

@author: Matt Whelan

The following is an attempt to replicate the results of Tatsuya Haga and Tomoki Fukai in their paper on reverse replay,
titled "Recurrent network model for learning goal-directed sequences through reverse replay".

This script contains the first rate-coded algorithm of Figure 1.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRecurrentNetwork(object):
    def __init__(self, no_cells=500, time_step=2.0, w_max=27, d=5):
        '''

        :param no_cells: integer, specifies number of cells in the linear network
        :param time_step: float, the step size used for simulation in milliseconds
        :param w_max: float, sets the max value for weight initialisation
        :param d: float, sets the standard deviation for the weights
        '''

        self.no_cells = no_cells
        self.time_step = time_step
        self.w_max = w_max
        self.d = d

        # Constants
        self.U = 0.6 # constant used in STP facilitation
        self.w_inh = 1.0

        # Initial conditions
        self.weights = self.initialise_weights(self.w_max, self.d)
        self.weights_initial = self.weights
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

        self.STP_F = np.ones(self.no_cells) * U
        self.STP_F_next = np.ones(self.no_cells) * U

    def initialise_weights(self, w_max, d):
        '''

        :param w_max: float, the max value for of weight
        :param d: float, the standard deviation used for setting the weights
        :return: numpy array of the weight matrix, where weights[i][j] specifies the weight between neurons i and j
        '''

        weights = np.zeros((self.no_cells, self.no_cells))
        for j in range(self.no_cells):
            for i in range(self.no_cells):
                if i == j:
                    weights[i][j] = 0.0
                else:
                    weights[i][j] = w_max * np.exp(-np.abs(i - j) / d)
        return weights

    def update_cell_firing_rates(self, I_ext, rho=0.0025, epsilon=0.5):
        '''

        :param I_ext: float, the external current input
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
            self.STP_F_next[f] = ((U - STP_F[f]) / tau_f + U * (1 - STP_F[f]) * rates[f]) * self.time_step + STP_F[f]
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


if __name__ == "__main__":
    network = LinearRecurrentNetwork()
