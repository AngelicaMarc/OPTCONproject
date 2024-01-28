import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import parameters as param
import cost as cst

# Import model parameters

ns = param.ns
ni = param.ni

dt = param.dt           # discretization step from dynamics
ts = param.num_steps    # number of time steps

tf = ts * dt            # Final time in seconds
tm = int(ts / 2)        # Middle time step

max_iters = 4

# Import equilibrium points

uu1 = np.zeros((ni))
uu2 = np.zeros((ni))
xx1 = np.zeros((ns))
xx2 = np.zeros((ns))

uu0 = np.zeros((ni))
uu0 = [0, 0, 0]


uu1 = [0.30953803, -1.04573824, -0.28607073]
xx1 = [600, 0.1, 0, 0]

uu2 = [0.32793345, -0.32531731, -0.14198654]
xx2 = [900, 0.2, 0, 0]

# Initialize the reference trajectory

traj_ref = np.zeros((ns+ni, ts))
traj_ref[:ns,0] = xx1
traj_ref[ns:,0] = uu1

print(f"initial state: {traj_ref[:ns,0]}")

for tt in range(1,ts):

    traj = param.dynamics(traj_ref[:ns,tt-1], traj_ref[ns:,tt-1],1)
    
    traj_ref[:ns, tt] = traj[:ns]  

    if tt < (ts/10):
        traj_ref[ns:, tt] = uu1

    else:  
        traj_ref[ns:, tt] = uu2


print(f"final state: {traj_ref[:ns,ts-1]}")

xx_ref = traj_ref[0:ns,:]
uu_ref = traj_ref[ns:,:]

# Plot of the reference trajectories
tt_hor = np.linspace(0,tf,ts)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, traj_ref[0,:], 'm--', linewidth=2)
axs[0].axhline(y=xx2[0], color='r', linestyle='--', linewidth=1)
axs[0].grid()
axs[0].set_ylabel('$V$', rotation=0)

axs[1].plot(tt_hor, traj_ref[1,:], 'm--', linewidth=2) 
axs[1].axhline(y=xx2[1], color='r', linestyle='--', linewidth=1)
axs[1].grid()
axs[1].set_ylabel('$\\alpha$', rotation=0)

axs[2].plot(tt_hor, traj_ref[2,:], 'm--', linewidth=2)
axs[2].axhline(y=xx2[2], color='r', linestyle='--', linewidth=1)
axs[2].grid()
axs[2].set_ylabel('$\\theta$', rotation=0)

axs[3].plot(tt_hor, traj_ref[3,:], 'm--', linewidth=2)
axs[3].axhline(y=xx2[3], color='r', linestyle='--', linewidth=1)
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