#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Plotting(object):
	def __init__(self):
		pass

	def plot_full_simulation(self):
		'''
		Plots the neuron rates and currents for the full 4s simulation
		:return: None
		'''

		rates_data = pd.DataFrame.from_csv('neuron_rates_4000.csv')
		currents_data = None
		print(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4))

		plt.pcolor(rates_data)
		plt.yticks([0, 100, 200, 300, 400, 500], [0, 100, 200, 300, 400, 500])
		plt.xticks(np.arange(0, len(rates_data.columns) + len(rates_data.columns) / 4, len(rates_data.columns) / 4), [0, 1, 2, 3, 4])
		plt.ylabel('Neuron #')
		plt.xlabel('Time (s)')
		plt.show()


if __name__ == "__main__":
	plots = Plotting()
	plots.plot_full_simulation()