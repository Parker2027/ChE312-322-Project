import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Tinit = 0 # Initial temperature of the ice spear
Tf0 = 5 # Initial temperature of the "water"
Tinf = 25 # Temperature of air in °C

steps = 6 # how many steps in time per hour
hr = 10 # how many hours to simulate

n = steps*hr - 1 # number of internal nodes

R1 = 3 # radius of ice spear # cm 
R2 = 5 # outer radius of system # cm
h = 20 # height of glass # cm
A1 = 2*np.pi*R1*h # area for convection between ice and "water" # cm2
A2 = 2*np.pi*R2*h # area for convection between "water" and air # cm2
V_liq = np.pi*(R2**2 - R1**2) * h # volume of "water" # cm3

k = 2.22/1000 # ice # W/(cm * K) 
hf = 1500/1000**2 # "water" # W/(cm2 * K)
ha = 500/1000**2 # air # W/(cm2 * K)

rho_ice = 0.916 # density of ice at 0°C # g/cm3
rho_water = 1 # density of "water" g/cm3
cp_ice = 2.050 # heat capacity of ice at 0°C #J/(g * K)
cp_water = 4.184 # J/(g * K) # heat capacity of "water"

alpha = k/(rho_ice*cp_ice) 

def dTdt(t, T0):
   '''
   Taking a list of initial temperatures of the nodes in the ice and the "water" temperature 
   and writting each internal node as a finite difference with radius
   '''
   T = T0[:-1] # Temperature at each node in the ice
   Tf = T0[-1] # "water" temperature

   delta_r = R1/(n+1)

   r = np.linspace(0, R1, n+2) # adding two for r =0 and r=R1 since n counts internal nodes
   dTdt = np.zeros(n+2)

   dTdt[0] = alpha * (2*T[1]-2*T[0])/(delta_r**2) # boundary condition at r = 0
   dTdt[1:-1] = alpha * ((T[2:]-2*T[1:-1]+T[:-2])/(delta_r**2) + 1/r[1:-1] * (T[2:]-T[:-2])/delta_r) # internal nodes with centered differences
   dTdt[-1] = -k/hf * (T[-1]-Tf) # boundary condition at r=R1, equal flux across the boundary

   dTfdt = -(ha * A2 * (Tf-Tinf) - hf * A1 * (T[-1]-Tf))/(rho_water*cp_water*V_liq) # diffential describing "water" temperature, based on energy balance

   return np.hstack([dTdt, dTfdt])

tend = 3600 * hr # time in seconds to evaluate
t_vals = np.linspace(0, tend, n+2) # 60 seconds per step

sol = solve_ivp(dTdt, (0, tend), np.hstack([np.ones(n+2)*Tinit, Tf0]), t_eval = t_vals)

r_vals = np.linspace(0, R1, n+2) # again plus two for the boundary nodes

tsteps = sol.t.size

solT = sol.y

plt.figure()
for i in range(0, tsteps):
    plt.cla()
    plt.ylim([0,30])
    plt.plot(r_vals,solT[:-1, i], label='t='+str(round(sol.t[i], 2)/60)+' minutes')
    plt.legend()
    plt.xlabel("Distance from Center (cm)")
    plt.ylabel("Temperature of Ice (°C)")
    plt.pause(0.01)
plt.show()

plt.plot(t_vals,solT[-1, :]) # e
plt.ylim([0,30]) # max temp is Tinf
plt.xlabel("t (s)")
plt.ylabel("Temperature of Liquid (°C)")
plt.show()

r, t = np.meshgrid(r_vals, t_vals) # r_vals, t_vals need to be the same size else plot_surface breaks
fig=plt.figure()
ax=plt.axes(projection='3d') # initiating the 3d plot
plot=ax.plot_surface(r,t,solT[:-1].T,cmap=plt.cm.coolwarm) #plotting ice temperature against distance from radius and time
ax.set_xlabel('r')
ax.set_ylabel('t')
ax.set_zlabel('T')
plt.show()
