#!/usr/bin/env python3

# Note that the smallest time constant we have in the equations is 10ms, for the current equations.
# Look what happens if the time step is only set to half of that (5ms) below. For a normal first order
# differential equation, the discrete time solution is wildly incorrect. Only for a time step that is around
# 20 times smaller than the time constant is the solution more acceptable.


import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0,60,0.01)

c = 5
y1 = 5*np.exp(-t/10)

t_step = 0.5
no_steps = int(60/t_step)
y2 = np.zeros(no_steps)
y2[0] = 5
for i in range(no_steps - 1):
    y2[i+1] = y2[i] + -0.1*y2[i] * t_step

plt.plot(t,y1)
plt.plot(np.arange(0,60,t_step),y2)
plt.show()
