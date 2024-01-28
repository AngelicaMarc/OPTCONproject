import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import parameters as param
import cost

# Import the number of states of the choosen dinamics

ns = param.ns
ni = param.ni

dt = param.dt           # discretization step from dynamics
ts = param.num_steps    # number of time steps

tf = ts * dt            # Final time in seconds
tm = int(ts / 2)        # Middle time step

# Import model parameters

VV = param.VV
alfa = param.alfa
JJ = param.JJ
gg = param.gg
rho = param.rho
CL = param.CL
CD = param.CD
CM = param.CM
CT = param.CT
BB = param.BB

# Import equilibrium points

uu1 = np.zeros((ni))
uu2 = np.zeros((ni))
xx1 = np.zeros((ns))
xx2 = np.zeros((ns))

uu1 = [0.37431072, -0.36268694, -0.14946046]
xx1 = [900, 0.1, 0, 0]
uu2 = [0.30953803, -1.04573824, -0.28607073]
xx2 = [600, 0.1, 0, 0]

# Initialize the reference trajectory

traj_ref = np.zeros((ns+ni, ts))
traj_ref[ns:,0] = uu1

for tt in range(1,ts):

    traj = param.dynamics(traj_ref[:ns,tt-1], traj_ref[ns:,tt-1],1)[0]
    print(traj)
    traj_ref[:ns, tt] = traj[:ns]  

    if tt < tm:
        traj_ref[ns:, tt] = uu1

    else:  
        traj_ref[ns:, tt] = uu2


xx_ref = traj_ref[0:ns,:]
uu_ref = traj_ref[ns:,:]

# Plot of the reference trajectories
tt_hor = np.linspace(0,tf,ts)

fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(tt_hor, traj_ref[0,:], 'm--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$V$')

axs[1].plot(tt_hor, traj_ref[1,:], 'm--', linewidth=2) 
axs[1].grid()
axs[1].set_ylabel('$\\alpha$')

axs[2].plot(tt_hor, traj_ref[2,:], 'm--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$\\theta$')

axs[3].plot(tt_hor, traj_ref[3,:], 'm--', linewidth=2)
axs[3].grid()
axs[3].set_ylabel('$q$')

axs[4].plot(tt_hor, traj_ref[4,:], 'm--', linewidth=2)
axs[4].grid()
axs[4].set_ylabel('$\delta_t$')

axs[5].plot(tt_hor, traj_ref[5,:], 'm--', linewidth=2)
axs[5].grid()
axs[5].set_ylabel('$\delta_c$')

axs[6].plot(tt_hor, traj_ref[6,:], 'm--', linewidth=2)
axs[6].grid()
axs[6].set_ylabel('$\delta_e$')
axs[6].set_xlabel('Time')

fig.suptitle("Reference")
fig.align_ylabels(axs)

plt.show()