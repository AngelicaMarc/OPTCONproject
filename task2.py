import numpy as np
import matplotlib.pyplot as plt
import parameters as param
import newton as nwt
import math

def sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

def custom_sigmoid(x, lower, upper, stretch_factor, translation_factor):
    scaled_x = (x - translation_factor) * stretch_factor
    sig = sigmoid(scaled_x)
    cust = lower + (upper - lower) * sig
    return cust

# Plot the equilibrium points
plot = 1
##############

# Import model parameters

ns = param.ns
ni = param.ni

dt = param.dt           # discretization step from dynamics
ts = param.num_steps    # number of time steps

tf = ts * dt            # Final time in seconds
tm = int(ts / 2)        # Middle time step

max_iters = 2

# Import equilibrium points

uu1 = np.zeros((ni))
uu2 = np.zeros((ni))
xx1 = np.zeros((ns))
xx2 = np.zeros((ns))

uu0 = np.zeros((ni))
uu0 = [0, 0, 0]

uu1 = [0.30953803, -1.04573824, -0.28607073]
xx1 = [600, 0.1, 0, 0]

uu2 = [0.42612887, -0.35995701, -0.14891448 ]
xx2 = [900, 0.1, 0.1, 0]

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
                traj_ref[ii, tt] = custom_sigmoid(tt, xx1[ii], xx2[ii], 0.001, tm)
            for jj in range(1, ni):
                traj_ref[4, tt] = custom_sigmoid(tt, uu1[0], uu2[0], 0.001, tm)
                traj_ref[5, tt] = custom_sigmoid(tt, uu1[1], uu2[1], 0.001, tm)
                traj_ref[6, tt] = custom_sigmoid(tt, uu1[2], uu2[2], 0.001, tm)


print(f"final state: {traj_ref[:ns,ts-1]}")

xx_ref = traj_ref[0:ns,:]
uu_ref = traj_ref[ns:,:]

if(plot):
    # Plot of the reference trajectories
    tt_hor = np.linspace(0,tf,ts)

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
# xx, uu, descent, JJ, kk = grad.Gradient(xx, uu, xx_ref, uu_ref, cst.QQt, cst.RRt, cst.QQT, max_iters)
xx, uu, descent, JJ, kk = nwt.Newton(xx, uu, xx_ref, uu_ref, x0, max_iters)

xx_star = xx[:,:,kk]
uu_star = uu[:,:,kk]
uu_star[:,-1] = uu_star[:,-2]        # for plotting purposes

# Plots of descent direction and cost

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

# Plotting the trajectory

plt.plot(xx_star[0,:]*np.cos(xx_star[2,:]-xx_star[1,:]), xx_star[0,:]*np.sin(xx_star[2,:]-xx_star[1,:]), label='Optimal Trajectory')
plt.plot(xx_ref[0,:]*np.cos(xx_ref[2,:]-xx_ref[1,:]), xx_ref[0,:]*np.sin(xx_ref[2,:]-xx_ref[1,:]),'m--', label='Reference Trajectory')
plt.title('Airplane Trajectory')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

