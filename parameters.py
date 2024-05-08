import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# This file contains the model parameters and the dynamics of the system
# It's also useful to verify equilibria and the Jacobian of the system

# Conditional variables
verify_equilibrium = 0
plot = 0
verify_jacobian = 0

# discretization step
dt = 1e-2

# Define time steps
num_steps = 1000
time = np.arange(0, num_steps * dt, dt)

# Model parameters
mm = 26.82
JJ = 595.9
gg = 32.17
rho = 0.0011
CL = 0.5
CD = 1.59
CM = 0.5
CT = 3.75
#a0 = 1016
BB = np.zeros((2, 2))
BB[0,0] = -3.575
BB[0,1] = 0.065
BB[1,0] = -1.3
BB[1,1] = 6.5


ns = 4
ni = 3

# Initial conditions
VV = 900 # longitudinal speed
alfa = 0.1 # angle of attack
theta = 0.1  # pitch
qq = 0  # pitch rate


xx = np.array([[VV], [alfa], [theta], [qq]])

xx = np.squeeze(xx)

# Control inputs
uu = np.zeros((ni,1))

uu[0] = 0.0
uu[1] = 0.0
uu[2] = 0.0

#uu[0] = CD / CT  # Throttle
#uu[1] = (mm*gg - CL + BB[1,1]*BB[0,1]*CM )/(1 - (BB[1,1]*BB[0,0])/(BB[1,0]*BB[0,1]))  # Canard
#uu[2] = (-CM - BB[0,0]*uu[1])/BB[0,1]  # Elevator

#uu = [0.30953803, -1.04573824, -0.28607073]
uu = np.squeeze(uu)

def dynamics(xx, uu, flag=1):
    
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
    xxp = np.zeros((ns,))
    
    VV, alfa, theta, qq = xx #state variables

    # print(VV)

    TT, CC, EE = uu #control inputs

    LL = 0.5 * CL * rho * VV**2 #Lift
    DD = 0.5 * CD * rho * VV**2 #Drag
    Ma = 0.5 * CM * rho * VV**2 #pitching moment

    Th = 0.5 * rho * VV**2 * CT * TT #Control Thrust
    L_delta = 0.5 * rho * VV**2 * (BB[0,0] * CC + BB[0,1] * EE) #Control Lift
    M_delta = 0.5 * rho * VV**2 * (BB[1,0] * CC + BB[1,1] * EE)  #Control Moment

    # Dynamics with the forward Euler method

    if flag:
        xxp[0] = VV + dt * (Th * np.cos(alfa) - DD - mm * gg * np.sin(theta - alfa)) / mm
        xxp[1] = alfa+ dt * (qq - (Th * np.sin(alfa) + LL + L_delta - mm * gg * np.cos(theta - alfa)) / (mm * VV))
        xxp[2] = theta + dt * qq
        xxp[3] = qq + dt * ((M_delta + Ma) / JJ)
    else:
        xxp[0] = dt * (Th* np.cos(alfa) - DD - mm * gg * np.sin(theta - alfa)) / mm
        xxp[1] = dt * (qq - (Th * np.sin(alfa) + LL + L_delta - mm * gg * np.cos(theta - alfa)) / (mm * VV))
        xxp[2] = dt * qq
        xxp[3] = dt * (M_delta + Ma) / JJ

    return xxp.squeeze()

def jacobian(xx, uu):
    
    VV, alfa, theta, qq = xx #state variables
    TT, CC, EE = uu #control inputs

    fu = np.zeros((ni, ns))
    fx = np.zeros((ns, ns))

    #df1
    fx[0,0] = 1 + dt * (rho * CT * TT * np.cos(alfa) * VV - CD * rho * VV) / mm
    fx[1,0] = dt * (-0.5 * rho * VV**2 * CT * TT * np.sin(alfa) + mm*gg*np.cos(theta-alfa)) / mm
    fx[2,0] = dt * (-gg*np.cos(theta-alfa))
    fx[3,0] = 0

    fu[0,0] = dt * (0.5 * VV**2 * rho * CT * np.cos(alfa)) / mm
    fu[1,0] = 0
    fu[2,0] = 0

    #df2
    fx[0,1] = -dt * (rho * CT * TT * np.sin(alfa) + CL * rho + rho * (BB[0,0] * CC + BB[0,1] * EE) + mm*gg*np.cos(alfa-theta)/VV**2) / (mm)
    fx[1,1] = 1 - dt * (0.5 * VV * rho * CT *TT * np.cos(alfa) - (1/VV)*mm*gg*np.sin(theta-alfa)) / (mm)
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
    fx[0,3] = dt * (rho * VV * CM + rho * VV * (BB[1,0] * CC + BB[1,1] * EE))/JJ
    fx[1,3] = 0
    fx[2,3] = 0
    fx[3,3] = 1

    fu[0,3] = 0
    fu[1,3] = dt * (0.5 * rho * VV**2 * BB[1,0]) / JJ
    fu[2,3] = dt * (0.5 * rho * VV**2 * BB[1,1]) / JJ

    return fx , fu

#def func1(xx):
    xx_full = np.zeros((ns,))
    xx_full[0] = VV
    xx_full[1] = alfa
    np.append(xx_full, xx,0)
    xx = dynamics(xx_full, uu, 0)
    return (xx[2:])

def func(input):
    result = dynamics(xx, input[:3], 0)
    return result 
    
def find_equilibria(u_guess):
    
    # Use fsolve to find the equilibria
    inputs = np.append(u_guess, 0.0)
    equilibrium_inputs = fsolve(func, inputs)
    return equilibrium_inputs

# Verify that the dynamics are zero at the equilibrium
if(verify_equilibrium):

    equilibrium_inputs = find_equilibria(uu)
    print(f"Equilibrium Inputs: {equilibrium_inputs}")
    uu = equilibrium_inputs[:3]

    # Initialize arrays to store state variables
    VV_values = np.zeros(num_steps)
    alfa_values = np.zeros(num_steps)
    theta_values = np.zeros(num_steps)
    qq_values = np.zeros(num_steps)
    Mach_values = np.zeros(num_steps)

    # Simulate dynamics over time

    print(f"Initial States: {xx}")
    ##
    for i in range(num_steps):
        VV_values[i] = xx[0]
        alfa_values[i] = xx[1]
        theta_values[i] = xx[2]
        qq_values[i] = xx[3]
        xx = dynamics(xx, uu, 1)
        #Mach_values[i] = xx[0] / a0
    ##
    print(f"Final States: {xx}")

    if(plot):
        # Plot state variables
        plt.figure(figsize=(10, 12))

        # Plot Longitudinal Speed
        plt.subplot(4, 1, 1)
        plt.plot(time, VV_values)
        plt.xlabel('Time')
        plt.ylabel('Longitudinal Speed')
        plt.grid(True)

        # Plot Angle of Attack
        plt.subplot(4, 1, 2)
        plt.plot(time, alfa_values)
        plt.xlabel('Time')
        plt.ylabel('Angle of Attack')
        plt.grid(True)

        # Plot Pitch
        plt.subplot(4, 1, 3)
        plt.plot(time, theta_values)
        plt.xlabel('Time')
        plt.ylabel('Pitch')
        plt.grid(True)

        # Plot Pitch Rate
        plt.subplot(4, 1, 4)
        plt.plot(time, qq_values)
        plt.xlabel('Time')
        plt.ylabel('Pitch Rate')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Verify that the Jacobian is correct
if(verify_jacobian):

    fx, fu = jacobian(xx, uu)
    fx = fx.T
    fu = fu.T
    xxp = dynamics(xx, uu)
    print(uu)
    # Verify the dynamics of the system
    np.testing.assert_allclose(xxp, fx @ xx + fu @ uu, atol=1e-03)
    print("Dynamics verification passed!")
