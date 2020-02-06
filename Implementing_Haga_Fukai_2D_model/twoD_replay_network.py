#!/usr/bin/env python3

'''
This script runs the 2D hippocampal model of Haga and Fukai. Its only class takes in a matrix of network weights for
a 50x50 network, along with some trajectory data (place cell activity data is optional, but it will be computed
seperately if not). The trajectory data should give an x and y coordinate position (in m), time step (in ms) and a
reward value. If the agent reaches a reward point, it stops for a given amount of time for a reverse replay event to
occur (which will simply involve external current to that point in the network). At the end of the trajectory data,
it then gives the final weight matrix.
'''

import numpy as np
import pandas as pd

class PlaceCells:

    def __init__(self, trajectory_df, network_weights=None, place_cell_activity=None):
        '''

        :param trajectory_df: pandas data frame of the trajectory data containing lists of the x and y coordinate
        positions (m), time steps (ms) and reward value.
        :param network_weights: numpy array, the network weight matrix with size 50x8.
        :param place_cell_activity: numpy array, each row in the array is a 2500 vector (flattened version of a 50x50
        array) of place cell activities for each time step. The number of rows should be the same size the the length
        of the trajectory data lists (i.e. same number of time steps).
        '''

        # Setting up the network to fit the maze
        self.no_cells = 50  # number of cells along one row of the maze
        self.total_no_cells = self.no_cells ** 2  # the total number of cells
        self.arena_size = np.array((50.0, 50.0))
        self.no_cells_per_m = np.size(np.zeros((self.no_cells, self.no_cells)), 0) / self.arena_size[0]

        # unpacking the trajectory data into lists
        self.t = trajectory_df['t'].tolist()
        self.trajectory_x = trajectory_df['pos_x'].tolist()
        self.trajectory_y = trajectory_df['pos_y'].tolist()
        self.rewards = trajectory_df['r'].tolist()
        self.sim_time = str(int(round(max(self.t) / 1000))) + 's'
        print(self.sim_time)

        # setting up the network weights if nothing is passed in
        if network_weights is not None:
            print("Network weights provided and will be used in the model")
            self.weights = network_weights
        else:
            print("No network weights provided", "\n", "Initialising a random weight matrix...")
            self.weights = self.initialise_weights(self.total_no_cells)
            np.save('data/initial_network_weights_' + self.sim_time + '.npy', self.weights)
            print("Complete")
        self.weights_next = np.zeros((self.total_no_cells, 8))

        # self.place_cell_activities forms the I_place currents. It is computed ahead of the learning computations to
        # save time
        if place_cell_activity is not None:
            print("Place cell activity provided for the trajectory and will be used for the simulation")
            self.place_cell_activities = place_cell_activity
        else:
            print("No place cell activity has been provided for the trajectory")
            print("Computing the trajectory's place cell activities...")
            self.place_cell_activities = self.compute_place_cell_activities(self.trajectory_x, self.trajectory_y,
                                                                            self.rewards)
            # Saving the place cell activities for convenience next time around
            np.save('data/place_cell_activity_testing_' + self.sim_time + '.npy', self.place_cell_activities)
            print("Complete. Saved the place cell activity as a numpy array with name 'place_cell_activity.npy'")

        # Constants
        self.U = 0.4  # constant used in STP facilitation
        self.w_inh = 0.0005

        # Network initial conditions
        self.x_pos = self.trajectory_x[0]
        self.y_pos = self.trajectory_y[0]

        self.delta_w = np.zeros((self.total_no_cells, 8))
        self.delta_w_next = np.zeros((self.total_no_cells, 8))

        self.rates = np.zeros(self.total_no_cells)
        self.rates_next = np.zeros(self.total_no_cells)

        self.I_E = np.zeros(self.total_no_cells)
        self.I_E_next = np.zeros(self.total_no_cells)

        self.I_inh = 0 # global inhibition is used
        self.I_inh_next = 0

        self.I_noise = np.zeros(self.total_no_cells)

        self.I_theta  = 0

        self.I_place = np.zeros(self.total_no_cells)

        self.STP_D = np.ones(self.total_no_cells)
        self.STP_D_next = np.ones(self.total_no_cells)

        self.STP_F = np.ones(self.total_no_cells) * self.U
        self.STP_F_next = np.ones(self.total_no_cells) * self.U

    def initialise_weights(self, total_no_cells):
        # weights are initially randomised, but made to obey the normalisation specification that
        # sum_{k,l} w_{i,j}^{k,l} = 0.5
        # In addition, as each cell is only connected to 8 others, it would be useless computing the
        # learning rates and activities across a 2500x2500 weight matrix
        # In weights[i,j], i represents the post-synapse and j the pre-synapse. I.e. for a given row of weights (say
        # weights[i]), the weights will be the incoming weights from its neighbouring pre-synapses
        weights = np.zeros((total_no_cells, 8))
        for i in range(total_no_cells):
            weights[i] = np.random.rand(8)
            weights[i] = weights[i] / sum(weights[i]) * 0.5
        return weights

    def normalise_weights(self, weights):
        '''
        Function to normalise the weights so that their sum equals 1 only if their sum is greater than one.
        :param weights: numpy array, 2500x8 array of the weights matrix
        :return: numpy array, 2500x8 array of the weights matrix, with each neuron's weight vector normalised as above
        '''

        normalised_weights = np.zeros((self.total_no_cells, 8))
        for i in range(self.total_no_cells):
            if sum(weights[i]) > 1:
                normalised_weights[i] = weights[i] / sum(weights[i])
            else:
                normalised_weights[i] = weights[i].copy()
        return normalised_weights

    def update_cell_firing_rates(self, I_E_next, rho=1, epsilon=0.002): # Equation 34
        '''
        # TEST complete
        :param I_E_next: numpy array, 2500x1 vector of the updated total excitatory currents
        :param rho: float, constant used for the linear rectifier unit
        :param epsilon: float, threshold in linear rectifier unit
        :return: numpy array, 2500x1 vector of the network rates
        '''

        rates_next = np.zeros(self.total_no_cells)
        for i in range(self.total_no_cells):
            if I_E_next[i] > epsilon:
                rates_next[i] = rho * (I_E_next[i] - epsilon)
            else:
                rates_next[i] = 0.0
        return rates_next

    def update_I_E(self, I_E, delta_t, weights, rates, STP_D, STP_F, I_inh, I_theta, I_place, I_noise): # Equation 33
        '''
        # TODO TEST
        :param I_E: numpy array, 2500x1 vector of the current network currents
        :param delta_t: float, time step in ms
        :param weights: the 2500x8 weight matrix
        :param rates: numpy array, 2500x1 rates vector
        :param STP_D: numpy array, 2500x1 STP_D vector
        :param STP_F: numpy array, 2500x1 STP_F vector
        :param I_inh: float, global network inhibition
        :param I_theta: float, global theta current
        :param I_place: numpy array, 25001x1 vector of place cell currents
        :param I_noise: numpy array, 25001x1 vector of network noise
        :return: numpy array, 2500x1 vector of the updated network currents
        '''

        tau = 10.0 # ms
        I_E_next = np.zeros(self.total_no_cells)
        for i in range(self.total_no_cells):
            sum_wrDF = 0
            for j in range(8):
                if self.is_computable(i,j):
                    neighbour_index = self.neighbour_index(i,j)
                    try:
                        sum_wrDF += weights[i][j] * rates[neighbour_index] \
                                * STP_D[neighbour_index] * STP_F[neighbour_index]
                    except IndexError:
                        print(i,j)
                else:
                    continue
            I_E_next[i] = (-I_E[i] / tau + sum_wrDF - I_inh - I_theta + I_place[i] + I_noise[i] ) * delta_t + I_E[i]
        return I_E_next

    def update_I_inh(self, I_inh, delta_t, rates, STP_F, STP_D): # Equation 35
        '''
        # TEST complete
        :param I_inh: float, global network inhibition
        :param delta_t: float, time step in ms
        :param rates: numpy array, 2500x1 rates vector
        :param STP_F: numpy array, 2500x1 STP_F vector
        :param STP_D: numpy array, 2500x1 STP_D vector
        :return: float, the updated I_inh
        '''

        tau_inh = 10 #ms
        sum_rDF = 0
        for i in range(self.total_no_cells):
            sum_rDF += rates[i] * STP_D[i] * STP_F[i]
        return (-I_inh / tau_inh + self.w_inh * sum_rDF ) * delta_t + I_inh

    def compute_place_cell_activities(self, trajectory_x, trajectory_y, reward_vals):
        '''

        :param trajectory_x: list, the trajectory data for x positions (m)
        :param trajectory_y: list, the trajectory data for y positions (m)
        :param reward_vals: list, the reward values at each time step. If reward[step] != 0, the agent should be
        resting and the C parameter set to 0.001 kHz
        :return: numpy array, each row in the array is a flattened version of a 50x50 matrix of the place cell
        activities at each time step. Has same number of rows as the length of trajectory_x (and trajectory_y)
        '''

        d = 2.0 # m
        no_cell_it = self.no_cells  # the number of cells along one row of the network
        cell_activities_array = np.zeros((len(trajectory_x), self.total_no_cells))
        progress = 0
        for k in range(len(trajectory_x)):
            progressed = (int(k/len(trajectory_x)*100) - progress != 0)
            progress = int(k/len(trajectory_x)*100)
            if k != 0:
                distance_change = (trajectory_x[k] - trajectory_x[k - 1]) + (trajectory_y[k] - trajectory_y[k - 1])
            else:
                distance_change = 0
            if distance_change != 0:
                C = 0.005
            elif distance_change == 0.0 and reward_vals[k] != 0.0: # this should run the reverse replays
                C = 0.001 # kHz
            else:
                continue
            cells_activity = np.zeros((50, 50))
            place = np.array((trajectory_x[k], trajectory_y[k]))
            for i in range(no_cell_it):
                for j in range(no_cell_it):
                    place_cell_field_location = np.array(((i / self.no_cells_per_m), (j / self.no_cells_per_m)))
                    cells_activity[i][j] = C * np.exp(
                        -1.0 / (2.0 * d ** 2.0) * np.dot((place - place_cell_field_location),
                                                         (place - place_cell_field_location)))
            cell_activities_array[k] = cells_activity.flatten()
            if progressed:
                print(int(k/len(trajectory_x)*100), '%', end="\r")
        return cell_activities_array

    def update_I_noise(self):
        return np.random.normal(0, 0.0005, 2500)

    def update_I_theta(self, t):
        '''
        # TEST complete
        :param t: float, the current time in ms
        :return: float, the global theta current
        '''

        B = 0.005 # kHz
        t_theta = 1000 / 7 # ms
        return B / 2 * (np.sin(2 * np.pi * t / t_theta) + 1)

    def update_STP(self, STP_F, STP_D, delta_t, rates): # Equations 36 and 37
        '''
        # TEST complete
        :param STP_F: numpy array, 2500x1 current STP_F vector
        :param STP_D: numpy array, 2500x1 current STP_D vector
        :param delta_t: float, time step (ms)
        :param rates: numpy array, 2500x1 of network rates
        :return: two numpy arrays, 2500x1 updated vectors of the STP variables
        '''

        tau_f = 200 # ms
        tau_d = 300 # ms
        STP_F_next = np.zeros(self.total_no_cells)
        STP_D_next = np.zeros(self.total_no_cells)
        for f in range(self.total_no_cells):
            STP_F_next[f] = ((self.U - STP_F[f]) / tau_f + self.U * (1 - STP_F[f]) * rates[f]) * delta_t + STP_F[f]
            STP_D_next[f] = ((1.0 - STP_D[f]) / tau_d - rates[f] * STP_D[f] * STP_F[f]) * delta_t + STP_D[f]
        return STP_F_next, STP_D_next

    def update_delta_w(self, delta_w, delta_t, rates, STP_F, STP_D): # Equation 41
        '''
        # TEST complete
        :param delta_w: numpy array, 2500x8 current delta_w matrix
        :param delta_t: float, time step (ms)
        :param rates: numpy array, 2500x1 vector of network rates
        :param STP_F: numpy array, 2500x1 vector of STP_F values
        :param STP_D: numpy array, 2500x1 vector of STP_D values
        :return: numpy array, 2500x8 updated delta_w matrix
        '''

        eta = 1.0
        tau_w = 30000.0 # ms
        delta_w_next = np.zeros((2500, 8))
        for i in range(self.total_no_cells):
            for j in range(8):
                if self.is_computable(i,j):
                    neighbour_index = self.neighbour_index(i,j)
                    delta_w_next[i][j] = delta_t * (- delta_w[i][j] + eta * rates[i]
                                                              * rates[neighbour_index] * STP_D[neighbour_index]
                                                              * STP_F[neighbour_index]) / tau_w + delta_w[i][j]
                else:
                    continue
        return delta_w_next

    def update_weights(self, weights, delta_t, delta_w): # Equation 40
        '''
        # TEST complete
        :param weights: numpy array, 2500x8 current weights matrix
        :param delta_w: numpy array, 2500x8 current delta_w values
        :param delta_t: float, time step (ms)
        :return: numpy array, 2500x8 updated weights matrix
        '''

        weights_next = np.zeros((2500, 8))
        for i in range(self.total_no_cells):
            for j in range(8):
                if self.is_computable(i,j):
                    weights_next[i][j] = delta_w[i][j] * delta_t + weights[i][j]
                else:
                    continue
        return weights_next

    def is_computable(self, i, j):
        '''
        # TEST complete
        :param i: integer, indicates which is the selected neuron
        :param j: integer, indicates which neighbour neuron i is receiving something from
        :return: bool
        '''

        # This is a confusing function, so I must describe it in detail. Because of the 2D arrangement of the
        # network, the neurons on the edges will not be connected to any neurons to its side (north most neurons have
        # no connections to its north, etc.). As a result, when performing the computations, such as computing the
        # incoming rates to a neuron from its neighbour neurons, it is worth first determining whether this
        # computation is valid (i.e. there is indeed a neighbour neuron in the specified position). This function
        # therefore determines whether a computation, whether it be incoming rates or updating weights, is valid or
        # not. For simplicity, this is computed for a 50x50 2D neural network.
        # It is important to note here that the order of connections is as follows: [W NW N NE E SE S SW]. So from j
        # = 0 to j = 7, the increment in the index represents a clockwise rotation starting at the point W.

        if i % 50 == 0 and (j == 0 or j == 1 or j == 7): # no W connections
            return False
        elif i in range(50) and (j == 1 or j == 2 or j == 3): # no N connections
            return False
        elif (i+1) % 50 == 0 and (j == 3 or j == 4 or j == 5): # no E connections
            return False
        elif i in range(2450, 2500) and (j == 5 or j == 6 or j == 7): # no S connections
            return False
        else: # it's a valid computation
            return True

    def neighbour_index(self, i, j):
        '''
        # TEST commplete
        :param i: integer, indicates which is the selected neuron
        :param j: integer, indicates which neighbour neuron i is receiving something from
        :return: bool
        '''

        # Due to the 2D structure of the network, it is important to find which index from the vector of neurons
        # should be used as the neighbour neuron. For instance, the 2D network is concatenated by each row. So the
        # first 50 neurons of the vector of neurons represents the first row of the 2D network. The next 50 represent
        # the second row, and so on. Hence, the connection that neuron i receives from its north will be located at
        # i-50. For simplicity, this is computed for a 50x50 2D neural network.
        # It is important to note here that the order of connections is as follows: [W NW N NE E SE S SW]. So from j
        # = 0 to j = 7, the increment in the index represents a clockwise rotation starting at the point W.

        if j == 0: # W connection
            return i - 1
        elif j == 1: # NW connection
            return i - 51
        elif j == 2: # N connection
            return i - 50
        elif j == 3: # NE connection
            return i - 49
        elif j == 4: # E connection
            return i + 1
        elif j == 5: # SE connection
            return i + 51
        elif j == 6: # S connection
            return i + 50
        elif j == 7: # SW connection
            return i + 49
        else:
            return IndexError

    def begin_simulation(self, t=None, trajectory_x=None, trajectory_y=None, rewards=None):
        '''

        :param t: List of floats, the time steps for the trajectory data.
        :param trajectory_x: List of the x coordinates for the trajectory. Has same length as t.
        :param trajectory_y: List of the y coordinates for the trajectory. Has same length as t.
        :param rewards: List of ints, indicates whether the agent has reached a reward or not. Has same length as t.
        '''

        if t is None:
            t = self.t
        if trajectory_x is None:
            trajectory_x = self.trajectory_x
        if trajectory_y is None:
            trajectory_y = self.trajectory_y
        if rewards is None:
            rewards = self.rewards

        network_firing_rates = np.zeros((len(t), 2500)) # for storing the rates of the neurons

        delta_t = 0.0
        progress = 0.0
        for step in range(len(t)):
            progressed = (round(step / len(t) * 100, 1) - progress != 0)
            progress = round(step / len(t) * 100, 1)
            self.x_pos = trajectory_x[step]
            self.y_pos = trajectory_y[step]

            self.I_place = self.place_cell_activities[step]
            if step != 0: # set the values at the previous time step to be equal to the ones for the current time step
                delta_t = t[step] - t[step-1]
                self.weights = self.weights_next.copy()
                self.delta_w = self.delta_w_next.copy()
                self.rates = self.rates_next.copy()
                self.I_E = self.I_E_next.copy()
                self.I_inh = self.I_inh_next
                self.STP_D = self.STP_D_next.copy()
                self.STP_F = self.STP_F_next.copy()

            # update all the network variables
            self.I_theta = self.update_I_theta(t[step])
            self.I_noise = self.update_I_noise()
            self.I_E_next = self.update_I_E(self.I_E, delta_t, self.weights, self.rates, self.STP_D, self.STP_F,
                                            self.I_inh, self.I_theta, self.I_place, self.I_noise)
            self.I_inh_next = self.update_I_inh(self.I_inh, delta_t, self.rates, self.STP_F, self.STP_D)
            self.STP_F_next, self.STP_D_next = self.update_STP(self.STP_F, self.STP_D, delta_t, self.rates)
            self.rates_next = self.update_cell_firing_rates(self.I_E_next)
            self.delta_w_next = self.update_delta_w(self.delta_w, delta_t, self.rates, self.STP_F, self.STP_D)
            self.weights_next = self.update_weights(self.weights, delta_t, self.delta_w)
            self.weights_next = self.normalise_weights(self.weights_next)

            if step == 0:
                print(sum(self.weights))

            network_firing_rates[step] = self.rates * 1000  # multiplied by 1000 for rate in units per second

            if progressed:
                print(int(step / len(t) * 100), "%", end="\r")
                # Useful for running on HPC so that the output file doesn't grow too large with the continuous print
                # statements. This will overwrite the previous line keeping the progress.txt file very small.
                with open("progress.txt", "w") as file:
                    file.write(str(progress) + "%")

        print(sum(self.weights))
        np.save('data/final_network_weights_' + self.sim_time + '.npy', self.weights)
        np.save('data/networks_rates_' + self.sim_time + '.npy', network_firing_rates)


if __name__ == '__main__':
    # trajectory_df = pd.read_csv("trajectory_data/original_1trials_2timestep.csv")
    # activities = np.load('data/place_cell_activity.npy')
    # network = PlaceCells(trajectory_df, place_cell_activity=activities)
    # network = PlaceCells(trajectory_df)
    # network.begin_simulation()
    #
    # testing
    trajectory_df = pd.read_csv("trajectory_data/original_1trials_2timestep.csv")
    activities = np.load('data/place_cell_activity_testing_15s.npy')
    network = PlaceCells(trajectory_df, place_cell_activity=activities)
