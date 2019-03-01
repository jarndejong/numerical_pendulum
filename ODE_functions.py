# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:32:29 2019

@author: Jarnd
"""
import numpy as np
from math import log10

###############################################################################
############################### Examples for fs ###############################
###############################################################################
def f_simple_pend(m,l,g):
    '''Computes the function f(y,t) = [f_0,f_1] in the system of 1st order DE y'(t) = f(y,t); with f_0 = y[1] and f_1 = -mgsin(f_0). '''
    def f(y,t):
        f_0 = y[1]
        f_1 = -1*m*g*np.sin(y[0])
        return np.array([f_0,f_1])
    return f

def f_simple_pend_lin(m,l,g):
    '''Computes the function f(y,t) = [f_0,f_1] in the system of 1st order DE y'(t) = f(y,t); with f_0 = y[1] and f_1 = -mgf_0. '''
    def f(y,t):
        f_0 = y[1]
        f_1 = -1*m*g*np.sin(y[0])
        return np.array([f_0,f_1])
    return f

def f_duffing_osc(alpha,beta,gamma,delta,omega):
    def f(y,t):
        f_0 = y[1]
        f_1 = gamma*np.cos(omega*t) - delta*y[1] - alpha * y[0] - beta * y[0] **3
        return np.array([f_0,f_1])
    return f

def functioncalculator(f,y,t):
    return f(y,t)
    
###############################################################################
################################# DE Solvers ##################################
###############################################################################
def solveDE_Euler(f,y_init,dt,nr_of_DE,t_tot):
    # Compute the total number of steps and the proper time that is a multiple of dt
    N_tot = int(t_tot/dt)
    
    # Make the time vector t
    t = [np.round(x*dt,np.int(-np.floor(log10(dt)))) for x in range(N_tot+1)]

    # Pre-allocate y
    y_all = np.zeros((nr_of_DE,N_tot+1))
    
    # Put the initial conditions in the solution
    y_all[:,0] = y_init
    
    # Run the actual algorithm
    for n in range(N_tot):    
        # Calculate y[n+1]
        y_all[:,n+1] = y_all[:,n] + dt*f(y_all[:,n],t[n])
    
    # Return the solution and the time vector
    return y_all, t

def solveDE_RK4(f,y_init,dt,nr_of_DE,t_tot):
    # Compute the total number of steps and the proper time that is a multiple of dt
    N_tot = int(t_tot/dt)
    
    # Make the time vector t
    t = [np.round(x*dt,np.int(-np.floor(log10(dt)))) for x in range(N_tot+1)]

    # Pre-allocate y
    y_all = np.zeros((nr_of_DE,N_tot+1))
    
    # Put the initial conditions in the solution
    y_all[:,0] = y_init
    
    # Run the actual algorithm
    for n in range(N_tot):
        # Calculate the different k's
        k1 = dt*f(y_all[:,n],t[n])
        k2 = dt*f(y_all[:,n]+k1/2,t[n]+dt/2)
        k3 = dt*f(y_all[:,n]+k2/2,t[n]+dt/2)
        k4 = dt*f(y_all[:,n]+k3,t[n]+dt)
    
        # Calculate y[n+1]
        y_all[:,n+1] = y_all[:,n] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Return the solution and the time vector
    return y_all, t
