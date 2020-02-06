#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation


class CreateAnimatedPlots:

	def __init__(self, t, pos_x, pos_y, reward, t_step, place_cell_activities=None, network_weights=None):
		'''

		:param t: List of floats, the time steps for the trajectory data.
		:param pos_x: List of the x coordinates for the trajectory. Has same length as t.
		:param pos_y: List of the y coordinates for the trajectory. Has same length as t.
		:param reward: List of bools, indicates whether the agent has reached a reward or not. Has same length as t.
		:param t_step: Int, the real time step for updating the plots in ms.
		:param place_cell_activities: List of numpy arrays, optional list of the place cell activities at each time step
		'''

		self.t = t
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.reward = reward
		self.t_step = t_step
		self.no_neurons = 2500 # must be a square number that fits a square arena (for convenience only)
		self.no_neurons_row = int(np.sqrt(self.no_neurons))

		if place_cell_activities is not None:
			self.network_activity_series = place_cell_activities
		else:
			self.cells_activity = np.zeros((50, 50))
			self.arena_size = np.array((50.0, 50.0))
			self.no_cells_per_m = np.size(self.cells_activity, 0) / self.arena_size[0]

			print("Computing list of place cell activities")
			self.network_activity_series = self.place_cell_computation()  # computes a list of the network place cell
			# activities over time
			concatenated_place_cell_activities = np.zeros((len(self.network_activity_series), self.no_neurons))
			for i, activities in enumerate(self.network_activity_series):
				concatenated_place_cell_activities[i] = np.concatenate(activities)
			np.save('place_cell_activity.npy', concatenated_place_cell_activities)
			print("Completed")
		self.iter_index = count()

		if network_weights is not None:
			self.network_weights = network_weights

	def begin_plotting(self):
		ani = FuncAnimation(plt.gcf(), self.plot_func, interval=self.t_step)
		plt.show()

	def plot_func(self, j):
		# This function is passed into the FuncAnimation class, which then provides the live plotting
		i = next(self.iter_index) * 100
		print(i)

		plt.figure(1, figsize=(5, 10))
		plt.subplot(2,1,1)
		plt.cla()
		plt.xlim((0, 50))
		plt.ylim((0, 50))
		plt.scatter(self.pos_x[i], self.pos_y[i])


		plt.subplot(2,1,2)
		plot = np.flip(np.transpose(self.network_activity_series[i]), 0)
		# # plot = np.transpose(self.network_activity_series[i])
		# # plot = self.network_activity_series[i]
		plt.cla()
		# plt.contourf(plot, cmap=plt.cm.hot)
		# plt.cla()
		plt.imshow(plot, cmap='hot', interpolation='nearest')
		# plt.colorbar()
		plt.draw()
		# plt.pause(0.1)

	def place_cell_computation(self):
		C = 5
		d = 2
		no_cell_it = 50 # the number of cells along one row of the network
		cell_activities_list = []
		for k in range(len(self.pos_x)):
			if k != 0:
				distance_change = (self.pos_x[k] - self.pos_x[k-1]) + (self.pos_y[k] - self.pos_y[k-1])
				if distance_change == 0.0:
					self.cells_activity = np.zeros((50, 50))
				else:
					self.cells_activity = np.zeros((50, 50))
					place = np.array((self.pos_x[k], self.pos_y[k]))
					for i in range(no_cell_it):
						for j in range(no_cell_it):
							place_cell_field_location = np.array(((i / self.no_cells_per_m), (j / self.no_cells_per_m)))
							self.cells_activity[i][j] = C * np.exp(
								-1.0 / (2.0 * d ** 2.0) * np.dot((place - place_cell_field_location),
								                                 (place - place_cell_field_location)))
				cell_activities_list.append(self.cells_activity)
		return cell_activities_list

	def plot_network_weights(self, network_weights):
		'''
		An optional function for plotting the vectors of the network weights.
		:param network_weights: numpy array, size total_no_cellsx8, representing the 8 neighbouring connections for
		each cell
		:return: None
		'''

		vector_vals = np.zeros((np.size(network_weights, 0), 2))
		for cell_no in range(np.size(network_weights, 0)):
			# recall that the directions for each of the 8 connections are ordered as [W NW N NE E SE S SW]
			vector_vals[cell_no, 0] = network_weights[cell_no,4] + network_weights[cell_no, 3] / np.sqrt(2) + \
			                          network_weights[cell_no,5] / np.sqrt(2) - network_weights[cell_no, 0] - \
			                          network_weights[cell_no,1] / np.sqrt(2) - network_weights[cell_no,7] / np.sqrt(2)
			vector_vals[cell_no, 1] = network_weights[cell_no,2] + network_weights[cell_no, 1] / np.sqrt(2) + \
			                          network_weights[cell_no,3] / np.sqrt(2) - network_weights[cell_no, 6] - \
			                          network_weights[cell_no,5] / np.sqrt(2) - network_weights[cell_no,7] / np.sqrt(2)

		# Now normalise the vectors so they all have length of one (easier then to visualise in the plot)
		normalised_weight_vecs = np.zeros((np.size(network_weights, 0), 2))
		for cell_no in range(np.size(network_weights, 0)):
			normalised_weight_vecs[cell_no] = vector_vals[cell_no] / np.linalg.norm(vector_vals[cell_no])
		print(normalised_weight_vecs[5])

		plt.figure(2)
		X = np.arange(0, 50, 1)
		Y = np.arange(0, 50, 1)
		U = normalised_weight_vecs[:,0].reshape((50,50))
		V = np.flip(normalised_weight_vecs[:,1].reshape((50,50)))
		plt.quiver(X, Y, U, V)
		plt.show()

if __name__ == "__main__":
	trajectory_df = pd.read_pickle("trajectory_data/original_4trials_2timestep.pkl")
	t = trajectory_df['t'].tolist()
	pos_x = trajectory_df['pos_x'].tolist()
	pos_y = trajectory_df['pos_y'].tolist()
	reward = trajectory_df['r'].tolist()
	# activities = np.load('place_cell_activity.npy')
	# activities = np.load('/home/matt/data/place_cell_activity_testing_90s.npy')
	activities = np.load('/home/matt/data/networks_rates_60s.npy')
	activity_list = []
	for i in range(len(activities)):
		activity_list.append(np.reshape(activities[i], (-1, 50)))
	plot = CreateAnimatedPlots(t, pos_x, pos_y, reward, 2, place_cell_activities=activity_list)
	# plot = CreateAnimatedPlots(t, pos_x, pos_y, reward, 2)
	plot.begin_plotting()

	test_network_weights = np.load('/home/matt/data/final_network_weights_60s.npy')
	print(test_network_weights[500])
	plot.plot_network_weights(test_network_weights)