# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:16:55 2019

@author: Jarnd
"""

# Import modules
import numpy as np
import scipy as sp
import numpy.linalg as lin
import matplotlib.pyplot as plt


#%% Define variables
# Timestep for the numerical simulation
N_tot = 50 # []

# Total length of simulation
t_tot = 5 # sec

# Mass of the top bob
m_1 = 0.3 # kg

# Length of top pendulum
l_1 = 0.5 # meter

# Gravitational acceleration
g = 9.81 # m/s^2
#%% Calculate other elements

# Calculate the length of each time step; maybe change into defining the time step and thereby calculating the N_tot
dt = t_tot/N_tot # sec


#%% Pre-allocate the time, angle and radial velocity vector
t = np.linspace(0,t_tot,N_tot)
th_1 = np.zeros_like(t)
v_1 = np.zeros_like(t)
a_1 = np.zeros_like(t)
#%% Define initial conditions
th_1[0] = 0.05 * np.pi # rad
v_1[0] = 0

#%% Algorithm for calculating th_1 and v_1
# We implement the DE: d^2(th)/dt^2 + m*g*sin(th) = 0

for n in range(N_tot-1):
    # First compute d^2(th)/dt^2 [n]:
    a_1[n] = -1*m_1*g*np.sin(th_1[n])
    
    # Calculate radial velocity at next timestep as v_new = v_old + a_old*dt (linearisation of v_old)
    v_1[n+1] = v_1[n] + dt*a_1[n]
    
    # Calculate angle at next timestep as th_new = th_old + v_old*dt (linearisation of th around th_old)
    th_1[n+1] = th_1[n] + dt*v_1[n]

#%% Plot some quick data for checking
fig1 = plt.figure()
plt.plot(t,th_1)