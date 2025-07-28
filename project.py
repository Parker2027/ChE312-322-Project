import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


Tinit = 0
Tf0 = 10

n = 18 # number of internal nodes
R1 = 3 # cm
R2 = 5 # cm
A1 = 2*np.pi*R1 # cm2
A2 = 2*np.pi*R2 # cm2

k = 2.22/1000 # J/(cm * K)
hf = 1500/1000**2 # J/(cm2 * K)
ha = 500/1000**2 # J/(cm2 * K)

rho = 1 # g/cm3
cp = 4.184 # J/(g * K)
alpha = k/(rho*cp)

Tinf = 25 # degrees celcius

def dTdt(t, T0):
   T = T0[:-1]
   Tf = T0[-1]
   delta_r = R1/(n+1)
   r = np.linspace(0, R1, n+2)
   dTdt = np.zeros(n+2)

   dTdt[0] = alpha * (2*T[1]-2*T[0])/(delta_r**2)
   dTdt[1:-1] = alpha * ((T[2:]-2*T[1:-1]+T[:-2])/(delta_r**2) + 1/r[1:-1] * (T[2:]-T[:-2])/delta_r)
   dTdt[-1] = -k/hf * (T[-1]-Tf)

   dTfdt = -(ha * A2 * (Tf-Tinf) - hf * A1 * (T[-1]-Tf))/(rho*cp)

   return np.hstack([dTdt, dTfdt])

tend = 3600
t_vals = np.linspace(0, tend, 301) # 12 seconds per step
sol = solve_ivp(dTdt, (0, tend), np.hstack([np.ones(n+2)*Tinit, Tf0]), t_eval = t_vals)

y_vals = np.linspace(0, R1, n+2)

tsteps = sol.t.size

soly = sol.y

plt.figure()
for i in range(0, tsteps, 5):
    plt.cla()
    plt.ylim([0,30])
    plt.plot(y_vals,soly[:-1, i], label='t='+str(round(sol.t[i], 2)/60)+' minutes')
    plt.legend()
    plt.xlabel("r (cm)")
    plt.ylabel("Temperature of Ice (Celcius)")
    plt.pause(0.01)
plt.show()


plt.plot(t_vals,soly[-1, :])
plt.ylim([0,30])
plt.xlabel("t (s)")
plt.ylabel("Temperature of Liquid (Celcius)")
plt.show()
