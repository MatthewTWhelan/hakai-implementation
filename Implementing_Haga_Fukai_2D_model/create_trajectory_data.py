#!/usr/bin/env python3
'''
Creates the trajectory data from Haga and Fukai (Figures 7 and 8)
'''

import numpy as np
import pandas as pd

no_trials = 4
time_step = 2 # ms
t_trial = 15000 # ms, time per trial
no_steps = int(t_trial / time_step)
data = {'t': [], 'pos_x': [], 'pos_y': [], 'r': []}
#t_series = np.arange(0, 15000, time_step)
#pos = np.zeros(((15000 / time_step), 2))
z_a = np.array((25, 15))
z_b = np.array((25, 35))
z_c1 = np.array((45, 35))
z_d1 = np.array((45, 15))
z_c2 = np.array((5, 35))
z_d2 = np.array((5, 15))

for trial in range(no_trials):
	if trial % 2 == 0:
		z_x = z_c1.copy()
		z_y = z_d1.copy()
	else:
		z_x = z_c2.copy()
		z_y = z_d2.copy()
	for step in range(no_steps):
		reward = 0
		t = step * time_step
		if t < 2000:
			pos = z_a
		elif t < 4000:
			pos = (t / 1000.0 - 2.0) * (z_b - z_a) / 2.0 + z_a
		elif t < 6000:
			pos = (t / 1000.0 - 4.0) * (z_x - z_b) / 2.0 + z_b
		elif t < 8000:
			pos = (t / 1000.0 - 6.0) * (z_y - z_x) / 2.0 + z_x
		else:
			pos = z_y
			if trial % 2 == 0:
				reward = 1
		data['t'].append(step * time_step + (trial * t_trial))
		data['pos_x'].append(pos[0])
		data['pos_y'].append(pos[1])
		data['r'].append(reward)

trajectory_df = pd.DataFrame.from_dict(data)
trajectory_df.to_pickle('trajectory_data/original_' + str(no_trials) + 'trials_' + str(time_step) + 'timestep.pkl')
pd.DataFrame.to_csv(trajectory_df, 'trajectory_data/original_' + str(no_trials) + 'trials_' + str(time_step) +
					'timestep.csv')

x_data = []
y_data = []
# for coord in data['pos']:
# 	x_data.append(coord[0])
# 	y_data.append(coord[1])


