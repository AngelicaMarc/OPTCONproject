import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib.image as mpimg
import parameters as param
import newton as nwt
import math
import cost as cst
import random
import solver_MPC


##############
plot = 1 
disturb = 1
max_iters = 10
##############

# Load the image
img = mpimg.imread('airplane_039.jpg')

# Import model parameters

ns = param.ns
ni = param.ni

dt = param.dt              # discretization step from dynamics
ts = param.num_steps       # number of time steps

tf = ts * dt               # Final time in seconds
tm = int(ts / 2)           # Middle time step
stretch = 2*dt*ts*0.001    # For the sigmoid to work properly

# Cost matrices
QQ = cst.QQt
RR = cst.RRt


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig
def custom_sigmoid(x, lower, upper, translation_factor, stretch_factor=stretch):
    scaled_x = (x - translation_factor) * stretch_factor
    sig = sigmoid(scaled_x)
    cust = lower + (upper - lower) * sig
    return cust

uu1 = np.zeros((ni))
uu2 = np.zeros((ni))
xx1 = np.zeros((ns))
xx2 = np.zeros((ns))
uu0 = np.zeros((ni))
uu0 = [0, 0, 0]

def func1(input):
    result = param.dynamics(xx1, input[:3], 0)
    return result 
def func2(input):
    result = param.dynamics(xx2, input[:3], 0)
    return result 
def find_equilibria(u_guess, type):
    # Use fsolve to find the equilibria
    inputs = np.append(u_guess, 0.0)
    if type == 1:
        equilibrium_inputs = fsolve(func1, inputs)
    else:
        equilibrium_inputs = fsolve(func2, inputs)
    eq = equilibrium_inputs[:3]
    return eq

xx1 = [600, 0.05, 0.1, 0]
uu1 = find_equilibria(uu0, 1)

xx2 = [900, 0.1, 0.2, 0]
uu2 = find_equilibria(uu0, 2)

# Initialize the reference trajectory

traj_ref = np.zeros((ns+ni, ts))
traj_ref[:ns,0] = xx1
traj_ref[ns:,0] = uu1

print(f"initial state: {traj_ref[:ns,0]}")
for tt in range(1,ts):
    # traj = param.dynamics(traj_ref[:ns,tt-1], traj_ref[ns:,tt-1],1)
    # traj_ref[:ns, tt] = traj[:ns]  
    if tt < ts/8:
        traj_ref[ns:, tt] = uu1
        traj_ref[:ns, tt] = xx1 
    else:
        if tt > 7*ts/8:
            traj_ref[ns:, tt] = uu2
            traj_ref[:ns, tt] = xx2
        else:
            for ii in range(0, ns-1):
                traj_ref[ii, tt] = custom_sigmoid(tt, xx1[ii], xx2[ii], tm)
            for jj in range(1, ni):
                traj_ref[4, tt] = custom_sigmoid(tt, uu1[0], uu2[0], tm)
                traj_ref[5, tt] = custom_sigmoid(tt, uu1[1], uu2[1], tm)
                traj_ref[6, tt] = custom_sigmoid(tt, uu1[2], uu2[2], tm)
print(f"final state: {traj_ref[:ns,ts-1]}")

xx_ref = traj_ref[0:ns,:]
uu_ref = traj_ref[ns:,:]

tt_hor = np.linspace(0,tf,ts)

if(plot):
    # Plot of the reference trajectories
    
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, traj_ref[0,:], color='b', linewidth=2)
    #axs[0].axhline(y=xx2[0], color='r', linestyle='--', linewidth=1)
    axs[0].grid()
    axs[0].set_ylabel('$V$', rotation=0)

    axs[1].plot(tt_hor, traj_ref[1,:], color='b', linewidth=2) 
    #axs[1].axhline(y=xx2[1], color='r', linestyle='--', linewidth=1)
    axs[1].grid()
    axs[1].set_ylabel('$\\alpha$', rotation=0)

    axs[2].plot(tt_hor, traj_ref[2,:], color='b', linewidth=2)
    #axs[2].axhline(y=xx2[2], color='r', linestyle='--', linewidth=1)
    axs[2].grid()
    axs[2].set_ylabel('$\\theta$', rotation=0)

    axs[3].plot(tt_hor, traj_ref[3,:], color='b', linewidth=2)
    #axs[3].axhline(y=xx2[3], color='r', linestyle='--', linewidth=1)
    axs[3].grid()
    axs[3].set_ylabel('$q$', rotation=0)

    axs[4].plot(tt_hor, traj_ref[4,:], color='r', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$\delta_t$', rotation=0)

    axs[5].plot(tt_hor, traj_ref[5,:], color='r', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$\delta_c$', rotation=0)

    axs[6].plot(tt_hor, traj_ref[6,:], color='r', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$\delta_e$', rotation=0)

    axs[6].set_xlabel('Time')

    fig.suptitle("Reference")
    fig.align_ylabels(axs)

    plt.show()
    
# Newton
xx = np.zeros((ns, ts, max_iters+1))   # state seq.
uu = np.zeros((ni, ts, max_iters+1))   # input seq.

# initial conditions
for tt in range(ts):
    xx[:,tt,0] = np.copy(xx_ref[:,0])
    uu[:,tt,0] = np.copy(uu_ref[:,0]) 

x0 = np.copy(xx_ref[:,0])
xx, uu, descent, JJ, kk = nwt.Newton(xx, uu, xx_ref, uu_ref, x0, max_iters)
print("Task 2 completed")

xx_star = xx[:,:,kk]
uu_star = uu[:,:,kk]
uu_star[:,-1] = uu_star[:,-2]        # for plotting purposes

# Plots of descent direction and cost
if(plot):
  plt.figure('descent direction')
  plt.plot(np.arange(kk), descent[:kk])
  plt.xlabel('$k$')
  plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
  plt.yscale('log')
  plt.grid()
  plt.show(block=False)

  plt.figure('cost')
  plt.plot(np.arange(kk), JJ[:kk])
  plt.xlabel('$k$')
  plt.ylabel('$J(\\mathbf{u}^k)$')
  plt.yscale('log')
  plt.grid()
  plt.show(block=False)

if(plot):
  fig, axs = plt.subplots(ns+ni, 1, sharex='all')

  axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
  axs[0].plot(tt_hor, xx_ref[0,:], 'm--', linewidth=2)
  axs[0].grid()
  axs[0].set_ylabel('$V$')

  axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
  axs[1].plot(tt_hor, xx_ref[1,:], 'm--', linewidth=2)
  axs[1].grid()
  axs[1].set_ylabel('$\\alpha$')

  axs[2].plot(tt_hor, xx_star[2,:], linewidth=2)
  axs[2].plot(tt_hor, xx_ref[2,:], 'm--', linewidth=2)
  axs[2].grid()
  axs[2].set_ylabel('$\\theta$')

  axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
  axs[3].plot(tt_hor, xx_ref[3,:], 'm--', linewidth=2)
  axs[3].grid()
  axs[3].set_ylabel('$q$')

  axs[4].plot(tt_hor, uu_star[0,:], linewidth=2)
  axs[4].plot(tt_hor, traj_ref[4,:], 'm--', linewidth=2)
  axs[4].grid()
  axs[4].set_ylabel('$\delta_t$')

  axs[5].plot(tt_hor, uu_star[1,:], linewidth=2)
  axs[5].plot(tt_hor, uu_ref[1,:], 'm--', linewidth=2)
  axs[5].grid()
  axs[5].set_ylabel('$\delta_c$')

  axs[6].plot(tt_hor, uu_star[2,:],'g', linewidth=2)
  axs[6].plot(tt_hor, uu_ref[2,:], 'm--', linewidth=2)
  axs[6].grid()
  axs[6].set_ylabel('$\delta_e$')

  axs[6].set_xlabel('time')
      
  plt.show()

  # Plotting the trajectory
  
# Define time interval
delta_t = param.dt  # for example 0.1 seconds

# Get the velocities in x and y
vx_star = xx_star[0, :] * np.cos(xx_star[2, :] - xx_star[1, :])
vy_star = xx_star[0, :] * np.sin(xx_star[2, :] - xx_star[1, :])
vx_ref = xx_ref[0, :] * np.cos(xx_ref[2, :] - xx_ref[1, :])
vy_ref = xx_ref[0, :] * np.sin(xx_ref[2, :] - xx_ref[1, :])

# Forward Euler: Integrate numerically the velocities to obtain the positions
x_star = np.cumsum(vx_star) * delta_t
y_star = np.cumsum(vy_star) * delta_t
x_ref = np.cumsum(vx_ref) * delta_t
y_ref = np.cumsum(vy_ref) * delta_t

if(plot):
  # Track trajectories
  plt.plot(x_star, y_star, label='Optimal Trajectory')
  plt.plot(x_ref, y_ref, 'm--', label='Reference Trajectory')
  plt.xlabel('X position')
  plt.ylabel('Y position')
  plt.legend()
  plt.title('Airplane Trajectories')
  plt.show()
  

# MPC - Task 4
Tsim = ts
A_opt = np.zeros((ns, ns, ts))
B_opt = np.zeros((ns, ni, ts))


########################
# Linear Dynamics - get nominal A,B matrices
########################
AAnom = np.zeros((ns,ns,ts))
BBnom = np.zeros((ns,ni,ts))

for tt in range(ts-1):
    fx, fu= param.jacobian(xx_star[:,tt], uu_star[:,tt])
    AAnom[:,:,tt] = fx.T
    BBnom[:,:,tt] = fu.T

#############################
# Model Predictive Control
#############################

T_pred = 10  # MPC Prediction horizon
Tsim = ts  # Simulation horizon

# Definition of the extended matrices
AA = np.zeros((ns, ns, ts + T_pred))
BB = np.zeros((ns, ni, ts + T_pred))
xx_opt = np.zeros((ns, ts + T_pred))
uu_opt = np.zeros((ni, ts + T_pred))

# Fill the extended matrices with the nominal matrices
AA[:, :, 0:ts] = AAnom
BB[:, :, 0:ts] = BBnom
xx_opt[:, 0:ts] = xx_star
uu_opt[:, 0:ts] = uu_star


# Extend matrices for prediction horizon without using a for loop
for tt in range(ts,ts+T_pred):
    AA[:,:,tt] = AAnom[:,:,-1]
    BB[:,:,tt] = BBnom[:,:,-1]
    xx_opt[:,tt] = xx_star[:,-1]
    uu_opt[:,tt] = uu_star[:,-1]

# Initialize arrays for real MPC simulation
xx_mpc = np.zeros((ns, Tsim))
uu_mpc = np.zeros((ni, Tsim))
xx_T_mpc = np.zeros((ns, T_pred, Tsim))

# Perturb initial conditions
def disturbance(x):
    n = len(x)
    print("Initial conditions: ", x)
    y = np.zeros((n))
    for i in range(n):
        y[i] = random.uniform(0.1, 0.2)
        x[i] = x[i] + x[i]*y[i]
    print("Disturbance: ", y)
    print("New initial conditions: ", x)
    return x

if(disturb):
  # initial conditions different from the ones of xx0_star 
  #xx_mpc[:,0] = disturbance(xx1)     
  # Else enter the initial conditions you want to test
  xx_mpc[:,0] = [650, 0.06, 0.11, 0]
else:
  xx_mpc[:,0] = xx_star[:,0]


for tt in range(Tsim-1): 
  xx_meas_t = xx_mpc[:,tt] 
  # Solve MPC problem - apply first input
  if tt%100 == 0: # print every 100 time instants
        print('MPC:\t t = {} /'.format(tt),Tsim)

  AA_temp = AA[:,:,tt:tt+T_pred] 
  BB_temp = BB[:,:,tt:tt+T_pred]
  xx_opt_temp = xx_opt[:,tt:tt+T_pred] 
  uu_opt_temp = uu_opt[:,tt:tt+T_pred]

  uu_mpc[:,tt], xx_T_mpc[:,:,tt] = solver_MPC.linear_mpc(AA_temp, BB_temp,QQ,RR,QQ, xx_meas_t, xx_opt_temp, uu_opt_temp, T_pred)[:2]
        
  xx_mpc[:,tt+1] = param.dynamics(xx_mpc[:,tt], uu_mpc[:,tt])
#######################################
# Plots
#######################################
uu_mpc[:,-1] = uu_mpc[:,-2]
uu_star[:,-1] = uu_star[:,-2]
time_values = np.arange(0, tf, dt) 

time = np.arange(Tsim)
fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(time, xx_mpc[0,:Tsim],'m', linewidth=2, label='MPC')
axs[0].plot(time, xx_star[0,:Tsim],'--g', linewidth=2, label='Optimal')
axs[0].grid()
axs[0].set_ylabel('$V$')
axs[0].set_xlim([-1,Tsim])

#####
axs[1].plot(time, xx_mpc[1,:Tsim],'m', linewidth=2, label='MPC')
axs[1].plot(time, xx_star[1,:Tsim], '--g', linewidth=2, label='Optimal')
axs[1].grid()
axs[1].set_ylabel('$alpha$')
axs[1].set_xlim([-1,Tsim])

#####
axs[2].plot(time, xx_mpc[2,:Tsim],'m', linewidth=2, label='MPC')
axs[2].plot(time, xx_star[2,:Tsim], '--g', linewidth=2, label='Optimal')
axs[2].grid()
axs[2].set_ylabel('$theta$')
axs[2].set_xlim([-1,Tsim])

#####
axs[3].plot(time, xx_mpc[3,:Tsim],'m', linewidth=2, label='MPC')
axs[3].plot(time, xx_star[3,:Tsim], '--g', linewidth=2, label='Optimal')
axs[3].grid()
axs[3].set_ylabel('$q$')
axs[3].set_xlim([-1,Tsim])

#####
axs[4].plot(time, uu_mpc[0,:Tsim],'m', linewidth=2, label='MPC')
axs[4].plot(time, uu_star[0,:Tsim],'--g', linewidth=2, label='Optimal')
axs[4].grid()
axs[4].set_ylabel('$delta_t$')
axs[4].set_xlim([-1,Tsim-T_pred])

#####
axs[5].plot(time, uu_mpc[1,:Tsim],'m', linewidth=2, label='MPC')
axs[5].plot(time, uu_star[1,:Tsim],'--g', linewidth=2, label='Optimal')
axs[5].grid()
axs[5].set_ylabel('$delta_c$')
axs[5].set_xlim([-1,Tsim])

#####
axs[6].plot(time, uu_mpc[2,:Tsim],'m', linewidth=2, label='MPC')
axs[6].plot(time, uu_star[2,:Tsim],'--g', linewidth=2, label='Optimal')
axs[6].grid()
axs[6].set_ylabel('$delta_e$')
axs[6].set_xlabel('time')
axs[6].set_xlim([-1,Tsim])

fig.align_ylabels(axs)

plt.legend()
plt.tight_layout()
plt.show()

# Define time interval
delta_t = param.dt  

# Get the velocities in x and y
vx_star = xx_star[0, :] * np.cos(xx_star[2, :] - xx_star[1, :])
vy_star = xx_star[0, :] * np.sin(xx_star[2, :] - xx_star[1, :])
vx_ref = xx_mpc[0, :] * np.cos(xx_mpc[2, :] - xx_mpc[1, :])
vy_ref = xx_mpc[0, :] * np.sin(xx_mpc[2, :] - xx_mpc[1, :])

# Forward Euler: Integrate numerically the velocities to obtain the positions
x_star = np.cumsum(vx_star) * delta_t
y_star = np.cumsum(vy_star) * delta_t
x_ref = np.cumsum(vx_ref) * delta_t
y_ref = np.cumsum(vy_ref) * delta_t

# Fix trajectory drift (only visual purposes)
y_ref = y_ref + (y_star[-1]-y_ref[-1])
x_ref = x_ref + (x_star[-1]-x_ref[-1])


print("Task 4 completed")
# Track trajectories
plt.plot(x_star, y_star, label='Optimal Trajectory')
plt.plot(x_ref, y_ref, 'm--', label='MPC Trajectory')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.title('Airplane Trajectories')
plt.show()

