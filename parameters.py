import numpy as np

# Model parameters
m = 26.82
J = 595.9
g = 32.17
rho = 0.0011
CL = 0.5
CD = 1.59
CM = 0.5
CT = 3.75
B = [[-3.575, 0.065], [-1.3, 6.5]]
a0 = 1016

# Initial conditions
V = 0.0  # longitudinal speed
alfa = 0.0  # angle of attack
theta = 0.0  # pitch
q = 0.0  # pitch rate

# Control inputs
dT = 0.0  # Throttle
dC = 0.0  # Canard
dE = 0.0  # Elevator



def dynamics(x, u):
    
    V, alfa, theta, q = x #state variables
    dT, dC, dE = u #control inputs

    L = 0.5 * CL * rho * V**2 #Lift
    D = 0.5 * CD * rho * V**2 #Drag
    Ma = 0.5 * CM * rho * V**2 #pitching moment

    T = 0.5 * rho * V**2 * CT * dT #Contro Thrust
    L_delta = 0.5 * rho * V**2 * B * dC #Control Lift
    M_delta = 0.5 * rho * V**2 * B * dE #Control Moment

    dxdt = np.zeros_like(x)

    dxdt[0]= (T * np.cos(alfa) - D - m * g * np.sin(theta - alfa)) / m
    dxdt[1] = q - (T * np.sin(alfa) + L + L_delta - m * g * np.cos(theta - alfa)) / (m * V)
    dxdt[2] = q
    dxdt[3] = (M_delta + Ma) / J    

    return dxdt