import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import parameters as param
import newton as nwt

# Plot the equilibrium points
plot = 0
intermediate_plots = 1
##############

# Import model parameters

ns = param.ns
ni = param.ni

dt = param.dt           # discretization step from dynamics
ts = param.num_steps    # number of time steps

tf = ts * dt            # Final time in seconds
tm = int(ts / 2)        # Middle time step

max_iters = 10
# Import equilibrium points

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

#uu1 = [0.19707068, -1.01877927, -0.28067893]
xx1 = [600, 0.1, 0.2, 0]
uu1 = find_equilibria(uu0, 1)


#uu2 = [0.42612887, -0.35995701, -0.14891448 ]
xx2 = [900, 0.1, 0.26, 0]
uu2 = find_equilibria(uu0, 2)

#uu2 = [0.28186975 -0.56941592 -0.19080626 ]
#xx2 = [750, 0.2, 0, 0]

# Initialize the reference trajectory

traj_ref = np.zeros((ns+ni, ts))
traj_ref[:ns,0] = xx1
traj_ref[ns:,0] = uu1

print(f"initial state: {traj_ref[:ns,0]}")

for tt in range(1,ts):

    #traj = param.dynamics(traj_ref[:ns,tt-1], traj_ref[ns:,tt-1],1)
    
    #traj_ref[:ns, tt] = traj[:ns]  

    if tt < tm:
        traj_ref[ns:, tt] = uu1
        traj_ref[:ns, tt] = xx1

    else:  
        traj_ref[ns:, tt] = uu2
        traj_ref[:ns, tt] = xx2


print(f"final state: {traj_ref[:ns,ts-1]}")

xx_ref = traj_ref[0:ns,:]
uu_ref = traj_ref[ns:,:]

tt_hor = np.linspace(0,tf,ts)

if(plot):
    # Plot of the reference trajectories

    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, traj_ref[0,:], 'm--', linewidth=2)
    #axs[0].axhline(y=xx2[0], color='r', linestyle='--', linewidth=1)
    axs[0].grid()
    axs[0].set_ylabel('$V$', rotation=0)

    axs[1].plot(tt_hor, traj_ref[1,:], 'm--', linewidth=2) 
    #axs[1].axhline(y=xx2[1], color='r', linestyle='--', linewidth=1)
    axs[1].grid()
    axs[1].set_ylabel('$\\alpha$', rotation=0)

    axs[2].plot(tt_hor, traj_ref[2,:], 'm--', linewidth=2)
    #axs[2].axhline(y=xx2[2], color='r', linestyle='--', linewidth=1)
    axs[2].grid()
    axs[2].set_ylabel('$\\theta$', rotation=0)

    axs[3].plot(tt_hor, traj_ref[3,:], 'm--', linewidth=2)
    #axs[3].axhline(y=xx2[3], color='r', linestyle='--', linewidth=1)
    axs[3].grid()
    axs[3].set_ylabel('$q$', rotation=0)

    axs[4].plot(tt_hor, traj_ref[4,:], 'm--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$\delta_t$', rotation=0)

    axs[5].plot(tt_hor, traj_ref[5,:], 'm--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$\delta_c$', rotation=0)

    axs[6].plot(tt_hor, traj_ref[6,:], 'm--', linewidth=2)
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

    ##############################################################
    # Design OPTIMAL TRAJECTORY  
    ##############################################################

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
    axs[4].plot(tt_hor, uu_ref[0,:], 'm--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$\delta_c$')

    axs[5].plot(tt_hor, uu_star[1,:], linewidth=2)
    axs[5].plot(tt_hor, uu_ref[1,:], 'm--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$\delta_m$')

    axs[6].plot(tt_hor, uu_star[2,:],'g', linewidth=2)
    axs[6].plot(tt_hor, uu_ref[2,:], 'm--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$\delta_e$')

    axs[6].set_xlabel('time')
        
    plt.show()

if(intermediate_plots):
  max_iters = kk
  for kk in range(max_iters):
    xx_star = xx[:,:,kk]
    uu_star = uu[:,:,kk]
    uu_star[:,-1] = uu_star[:,-2]    

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
    axs[4].set_ylabel('$\delta_c$')  
    axs[5].plot(tt_hor, uu_star[1,:], linewidth=2)
    axs[5].plot(tt_hor, uu_ref[1,:], 'm--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$\delta_m$')  
    axs[6].plot(tt_hor, uu_star[2,:],'g', linewidth=2)
    axs[6].plot(tt_hor, uu_ref[2,:], 'm--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$\delta_e$')  
    axs[6].set_xlabel('time')
      
    plt.show()

# Plotting the trajectory
# plt.plot(xx_star[0,:]*np.cos(xx_star[2,:]-xx_star[1,:]), xx_star[0,:]*np.sin(xx_star[2,:]-xx_star[1,:]), label='Optimal Trajectory')
# plt.plot(xx_ref[0,:]*np.cos(xx_ref[2,:]-xx_ref[1,:]), xx_ref[0,:]*np.sin(xx_ref[2,:]-xx_ref[1,:]),'m--', label='Reference Trajectory')
# plt.title('Airplane Trajectory')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()

# Define time interval
delta_t = param.dt  # for example 0.1 seconds

# Get the velocities in x and y
vx_star = xx_star[0, :] * np.cos(xx_star[2, :] - xx_star[1, :])
vy_star = xx_star[0, :] * np.sin(xx_star[2, :] - xx_star[1, :])
vx_ref = xx_ref[0, :] * np.cos(xx_ref[2, :] - xx_ref[1, :])
vy_ref = xx_ref[0, :] * np.sin(xx_ref[2, :] - xx_ref[1, :])

# Forward Euler: Integrate numerically the velocities to obtain the positions
x_star = np.zeros_like(vx_star)
x_star[0] = 0
for i in range(1, len(vx_star)):
    x_star[i] = x_star[i-1] + vx_star[i] * delta_t

y_star = np.zeros_like(vy_star)
y_star[0] = 0
for i in range(1, len(vy_star)):
    y_star[i] = y_star[i-1] + vy_star[i] * delta_t

x_ref = np.zeros_like(vx_ref)
x_ref[0] = 0
for i in range(1, len(vx_ref)):
    x_ref[i] = x_ref[i-1] + vx_ref[i] * delta_t

y_ref = np.zeros_like(vy_ref)
y_ref[0] = 0
for i in range(1, len(vy_ref)):
    y_ref[i] = y_ref[i-1] + vy_ref[i] * delta_t

# Track trajectories
if(plot):
    plt.plot(x_star, y_star, label='Optimal Trajectory')
    plt.plot(x_ref, y_ref, 'm--', label='Reference Trajectory')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.title('Airplane Trajectories')
    plt.show()


