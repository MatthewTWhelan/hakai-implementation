#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Plotting(object):
	def __init__(self):
		pass

	def plot_full_simulation(self):
		rates_data = pd.DataFrame.from_csv('full_sim_rates_data.csv')
		currents_data = None
		weights_data = None

		plt.pcolor(rates_data)
		plt.yticks(np.arange(1, len(rates_data.index), 1), rates_data.index)
		plt.xticks(np.arange(1, len(rates_data.columns), 1), rates_data.columns)
		plt.show()


if __name__ == "__main__":
	plots = Plotting()
	plots.plot_full_simulation()