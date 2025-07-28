import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

alpha = 5
Tinit = 0

n = 5
R1 = 3
R2 = 5
k = 1
hi = 1
ha = 1

Tf = 10

def dTdt(t, T):
   delta_r = R1/(n+1)
   r = np.linspace(0, R1, n+2)
   dTdt = np.zeros(n+2)
   dTdt[0] = alpha * (2*T[1]-2*T[0])/(delta_r**2)
   #print(np.shape(dTdt[1:-1]), np.shape(T[2:]), np.shape(T[:-2]), np.shape(r[1:-1]), np.shape(T[2:]), np.shape(T[:-2]))
   dTdt[1:-1] = alpha * ((T[2:]-2*T[1:-1]+T[:-2])/(delta_r**2) + 1/r[1:-1] * (T[2:]-T[:-2])/delta_r)
   dTdt[-1] = -k/hi * (T[-1]-Tf)

   return dTdt

tend = 5

sol = solve_ivp(dTdt, (0, tend), np.ones(n+2)*Tinit)

y_vals = np.linspace(0, 10, n+2)

tsteps = sol.t.size

soly = sol.y

plt.figure()
for i in range(0, tsteps, 2):
	plt.cla()
	plt.ylim([0,10])
	plt.plot(y_vals,soly[:, i], label='t='+str(round(sol.t[i], 2))+'seconds')
	plt.legend()
	plt.xlabel("r (cm)")
	plt.ylabel("Temperature (Celcius)")
	plt.pause(0.01)
plt.show()
