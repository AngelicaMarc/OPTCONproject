import numpy as np
import matplotlib.pyplot as plt

# Model parameters
mm = 26.82
JJ = 595.9
gg = 32.17
rho = 0.0011
CL = 0.5
CD = 1.59
CM = 0.5
CT = 3.75
a0 = 1016
BB = np.zeros((2, 2))
BB[0,0] = -3.575
BB[0,1] = 0.065
BB[1,0] = -1.3
BB[1,1] = 6.5


ns = 4
ni = 3

# Initial conditions
VV = 1500  # longitudinal speed
alfa = 0  # angle of attack
theta = 0  # pitch
qq = 20  # pitch rate

xx = np.array([[VV], [alfa], [theta], [qq]])

# Control inputs
uu = np.zeros((ni,1))
uu[0] = 0.0  # Throttle
uu[1] = 0.0  # Canard
uu[2] = 0.0  # Elevator

# discretization step
dt = 1e-3

def dynamics(xx, uu):
    
    # State
    # xx[0] = VV
    # xx[1] = alfa
    # xx[2] = theta
    # xx[3] = qq

    # Input
    # uu[0] = TT
    # uu[1] = CC
    # uu[2] = EE

    
    # Initialization
    xxp = np.zeros((ns,1))
    
    VV, alfa, theta, qq = xx #state variables

    LL = 0.5 * CL * rho * VV**2 #Lift
    DD = 0.5 * CD * rho * VV**2 #Drag
    Ma = 0.5 * CM * rho * VV**2 #pitching moment

    Th = 0.5 * rho * VV**2 * CT * uu[0] #Control Thrust
    L_delta = 0.5 * rho * VV**2 * (BB[0,0] * uu[1] + BB[0,1] * uu[2]) #Control Lift
    M_delta = 0.5 * rho * VV**2 * (BB[1,0] * uu[1] + BB[1,1] * uu[2])  #Control Moment

    xxp[0] = xx[0,0] + dt * (Th * np.cos(alfa) - DD - mm * gg * np.sin(theta - alfa)) / mm
    xxp[1] = xx[1,0] + dt * (qq - (Th * np.sin(alfa) + LL + L_delta - mm * gg * np.cos(theta - alfa)) / (mm * VV))
    xxp[2] = xx[2,0] + dt * qq
    xxp[3] = xx[3,0] + dt * (M_delta + Ma) / JJ 

    fu = np.zeros((ni, ns))
    fx = np.zeros((ns, ns))

    #df1
    fx[0,0] = 1 + dt * (rho * CT * uu[0] * np.cos(alfa) * VV - CD * rho * VV) / mm
    fx[1,0] = dt * (-0.5 * rho * VV**2 * CT * uu[0] * np.sin(alfa) + mm*gg*np.cos(theta-alfa)) / mm
    fx[2,0] = dt * (-gg*np.cos(theta-alfa))
    fx[3,0] = 0

    fu[0,0] = dt * (0.5 * VV**2 * rho * CT * np.cos(alfa)) / mm
    fu[1,0] = 0
    fu[2,0] = 0

    #df2
    fx[0,1] = dt * (rho * CT * uu[0] * np.sin(alfa) + CL * rho + rho * (BB[0,0] * uu[1] + BB[0,1] * uu[2]) + mm*gg*np.cos(alfa)*VV**2) / (2*mm)
    fx[1,1] = 1 - dt * (-0.5 * VV * rho * CT *uu[0] * np.cos(alfa) - (1/VV)*mm*gg*np.sin(theta-alfa)) / (mm)
    fx[2,1] = dt * (-gg*np.sin(theta-alfa)) / (VV)
    fx[3,1] = dt

    fu[0,1] = dt * (-0.5 * VV * rho * CT * np.sin(alfa)) / (mm)
    fu[1,1] = dt * (-0.5 * rho * VV * (BB[0,0])) / mm
    fu[2,1] = dt * (-0.5 * rho * VV * (BB[0,1])) / mm

    #df3
    fx[0,2] = 0
    fx[1,2] = 0
    fx[2,2] = 1
    fx[3,2] = dt

    fu[0,2] = 0
    fu[1,2] = 0
    fu[2,2] = 0

    #df4
    fx[0,3] = dt * (rho * VV * CM + rho * VV * (BB[1,0] * uu[1] + BB[1,1] * uu[2]))/JJ
    fx[1,3] = 0
    fx[2,3] = 0
    fx[3,3] = 1

    fu[0,3] = 0
    fu[1,3] = dt * (0.5 * rho * VV * BB[1,0]) / JJ
    fu[2,3] = dt * (0.5 * rho * VV * BB[1,1]) / JJ

    return xxp

xx=dynamics(xx,uu)

# Define time steps
num_steps = 10000
time = np.arange(0, num_steps * dt, dt)

# Initialize arrays to store state variables
VV_values = np.zeros(num_steps)
alfa_values = np.zeros(num_steps)
theta_values = np.zeros(num_steps)
qq_values = np.zeros(num_steps)
Mach_values = np.zeros(num_steps)

# Simulate dynamics over time
for i in range(num_steps):
    xx = dynamics(xx, uu)
    VV_values[i] = xx[0]/100
    alfa_values[i] = xx[1]
    theta_values[i] = xx[2]
    qq_values[i] = xx[3]
    Mach_values[i] = xx[0] / a0

# Plot state variables
plt.figure(figsize=(10, 6))
plt.plot(time, VV_values, label='Longitudinal Speed')
plt.plot(time, alfa_values, label='Angle of Attack')
plt.plot(time, theta_values, label='Pitch')
plt.plot(time, qq_values, label='Pitch Rate')
plt.plot(time, Mach_values, label='Mach Number')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend()
plt.grid(True)
plt.show()
