#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Plotting(object):
	def __init__(self):
		pass

	def plot_full_simulation(self):
		'''
		Plots the neuron rates for the full 4s simulation
		:return: None
		'''

		rates_data = pd.DataFrame.from_csv('neuron_rates_4000.csv')
		print(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4))

		plt.pcolor(rates_data)
		plt.yticks([0, 100, 200, 300, 400, 500], [0, 100, 200, 300, 400, 500])
		plt.xticks(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4), [0, 1, 2, 3, 4])
		plt.ylabel('Neuron #')
		plt.xlabel('Time (s)')
		plt.show()

	def plot_middle_weight_3s_simulation(self):
		'''
		Plots the weights between the middle neuron at the rest at the 3s mark
		:return:
		'''

		rates_data = pd.DataFrame.from_csv('neuron_rates_3000.csv')
		print(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4))

		plt.pcolor(rates_data)
		plt.yticks([0, 100, 200, 300, 400, 500], [0, 100, 200, 300, 400, 500])
		plt.xticks(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4), [0, 1, 2, 3])
		plt.ylabel('Neuron #')
		plt.xlabel('Time (s)')
		plt.show()

	def plot_custom_simulation(self):
		'''
		Plots the neuron rates. Change the parameters as required.
		:return: None
		'''

		rates_data = pd.DataFrame.from_csv('neuron_rates_100.csv')
		print(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4))

		plt.pcolor(rates_data)
		plt.yticks([0, 100, 200, 300, 400, 500], [0, 100, 200, 300, 400, 500])
		plt.ylabel('Neuron #')
		plt.xlabel('Time (s)')
		plt.show()


if __name__ == "__main__":
	plots = Plotting()
	plots.plot_full_simulation()
	#plots.plot_custom_simulation()
	#plots.plot_middle_weight_3s_simulation()