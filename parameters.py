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
VV = 10  # longitudinal speed
alfa = 0.4  # angle of attack
theta = 0.1  # pitch
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

    return xxp

xx=dynamics(xx,uu)

# Define time steps
num_steps = 1000
time = np.arange(0, num_steps * dt, dt)

# Initialize arrays to store state variables
VV_values = np.zeros(num_steps)
alfa_values = np.zeros(num_steps)
theta_values = np.zeros(num_steps)
qq_values = np.zeros(num_steps)

# Simulate dynamics over time
for i in range(num_steps):
    xx = dynamics(xx, uu)
    VV_values[i] = xx[0]
    alfa_values[i] = xx[1]
    theta_values[i] = xx[2]
    qq_values[i] = xx[3]

# Plot state variables
plt.figure(figsize=(10, 6))
plt.plot(time, VV_values, label='Longitudinal Speed')
plt.plot(time, alfa_values, label='Angle of Attack')
plt.plot(time, theta_values, label='Pitch')
plt.plot(time, qq_values, label='Pitch Rate')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend()
plt.grid(True)
plt.show()

# TODO: compute the gradient of the dynamics
# fx = np.zeros((ns, ns))
# fu = np.zeros((ni, ns))
#######################
#########àà