import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

alpha = 0.05
Tinit = 0

def dTdt(t, T):

    n = 9
    R1 = 1

    delta_r = R1/(n+1)
    r = np.linspace(0.00000001, R1, n+1)

    T_ice = np.hstack([0, T, 10])

    dTdt = np.zeros(n+2)

    dTdt[0] = alpha * ((2*T[1]-2*T[0])/(delta_r**2) + 1/r[0] * T_ice[1]/delta_r)

    for i in range(1, n+1):
        dTdt[i] = alpha * ((T_ice[i+1]-2*T[i]+T_ice[i-1])/(delta_r**2) + 1/r[i] * (T_ice[i+1]-T_ice[i-1])/(2*delta_r))

    dTdt[-1] = alpha* ((2*T_ice[-2]-2*T[-1])/(delta_r**2) + 1/R1 * (T_ice[-2])/(delta_r))
    
    return dTdt

tend = 100

sol = solve_ivp(dTdt, (0, tend), np.ones(9)*Tinit)
print(sol.y)

