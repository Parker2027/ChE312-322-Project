import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


Tinit = 0
Tf0 = 10

n = 5
R1 = 3
R2 = 5
A1 = 2*np.pi*R1
A2 = 2*np.pi*R2

k = 0.22
hf = 1
ha = 1

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

tend = 10

sol = solve_ivp(dTdt, (0, tend), np.hstack([np.ones(n+2)*Tinit, Tf0]))

y_vals = np.linspace(0, 10, n+2)

tsteps = sol.t.size

soly = sol.y

plt.figure()
for i in range(0, tsteps, 2):
	plt.cla()
	plt.ylim([0,25])
	plt.plot(y_vals,soly[:-1, i], label='t='+str(round(sol.t[i], 2))+'seconds')
	plt.legend()
	plt.xlabel("r (cm)")
	plt.ylabel("Temperature (Celcius)")
	plt.pause(0.01)
plt.show()
