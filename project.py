import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

alpha = 1
Tinit = 0

def dTdt(t, T):
    n = 9
    R1 = 1

    delta_r = R1/(n+1)
    r = np.linspace(0, R1, n)

    T_ice = np.hstack([Tinit, T])

    for i in range(1, n+1):
        dTdt = alpha/(delta_r**2) * ((T_ice[i+1]-2*T[i]+T_ice[i-1]) + 1/r[i] * (T_ice[i+1]-T[i-1]))

    return dTdt

    
