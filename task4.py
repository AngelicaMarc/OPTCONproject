import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib.image as mpimg
import parameters as param
import newton as nwt
import math
import cost as cst
import cvxpy as cp
import random


##############
plot = 0
max_iters = 15
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

QQ = np.diag([0.001, 10000, 100, 0.001])     
RR = np.diag([1.0, 100.0, 1.0]) # TO FIX alpha weight
QQf = cst.QQT


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

#uu1 = [0.30953803, -1.04573824, -0.28607073]
xx1 = [600, 0.1, 0, 0]
uu1 = find_equilibria(uu0, 1)

#uu2 = [0.42612887, -0.35995701, -0.14891448 ]
xx2 = [900, 0.1, 0.06, 0]
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

# MPC - Task 4
Tsim = ts
A_opt = np.zeros((ns, ns, ts))
B_opt = np.zeros((ns, ni, ts))

def linear_mpc(AA, BB, QQ, RR, tl, QQf, xxt, thetamin, T_pred):

  xxt = xxt.squeeze()
  
  xx_mpc = cp.Variable((ns, T_pred))
  uu_mpc = cp.Variable((ni, T_pred))

  cost = 0
  constr = []
  # Tsim-1-T_pred
  for tt in range(tl, tl + T_pred -1):
    cost += cp.quad_form(xx_mpc[:,tt-tl] - xx_star[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt-tl] - uu_star[:,tt], RR)
    constr += [xx_mpc[:,tt+1-tl] == AA[:,:,tt]@xx_mpc[:,tt-tl] + BB[:,:,tt]@uu_mpc[:,tt-tl],  # dynamics constraint
            # other max/min values constraint
            # xx_mpc[0,tt+1-tl] >= 508, xx_mpc[0,tt+1-tl] <= 2032,  # V
            ]

  # sums problem objectives and concatenates constraints
  cost += cp.quad_form(xx_mpc[:,T_pred-1] - xx_star[:,tl+T_pred-1], QQf)
  constr += [xx_mpc[:,0] == xxt]

  problem = cp.Problem(cp.Minimize(cost), constr)
  problem.solve()

  if problem.status == "infeasible":
  # Otherwise, problem.value is inf or -inf, respectively
    print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

  return xx_mpc.value, uu_mpc.value

#############################
# Model Predictive Control
#############################

T_pred = 5      # MPC Prediction horizon
thetamin = -0.1

xx_real_mpc = np.zeros((ns,Tsim))
uu_real_mpc = np.zeros((ni,Tsim))

xx_mpc = np.zeros((ns, T_pred, Tsim))
uu_mpc = np.zeros((ni, T_pred, Tsim))

def disturbance(x):
    y = np.zeros((4))
    for i in range(4):
        y[i] = random.uniform(-0.1, 0.1)
    return x + x*y

xx_real_mpc[:,0] = disturbance(xx1)    # initial conditions different from the ones of xx0_star 

for tt in range(Tsim-1):
  # System evolution - real with MPC

  xx_t_mpc = xx_real_mpc[:,tt]  # get initial condition
  
  fx, fu = param.jacobian(xx_star[:,tt], uu_star[:,tt])

  A_opt[:,:,tt] = fx.T
  B_opt[:,:,tt] = fu.T

  # Solve MPC problem - apply first input

  if tt%10 == 0: # print every 10 time instants
    print('MPC:\t t = {:.1f} sec.'.format(tt*dt))

  if tt < Tsim-T_pred:
    xx_mpc[:,:,tt], uu_mpc[:,:,tt]  = linear_mpc(A_opt, B_opt, QQ, RR, tt, cst.QQT, xx_t_mpc, thetamin, T_pred = T_pred)
    
    uu_real_mpc[:,tt] = uu_mpc[:,0,tt]
    xx_real_mpc[:,tt+1] = param.dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt])

  else:
    #TO FIX
    uu_real_mpc[:,tt] = uu_mpc[:,tt-(Tsim-T_pred),Tsim-T_pred-1]
    xx_real_mpc[:,tt+1] = param.dynamics(xx_real_mpc[:,Tsim-T_pred-1], uu_real_mpc[:,Tsim-T_pred-1])

uu_real_mpc[:,-1] = uu_real_mpc[:,-2]        # for plotting purposes
#######################################
# Plots
#######################################
#print(xx_real_mpc)

time = np.arange(Tsim)
fig, axs = plt.subplots(ns+ni, 1, sharex='all')

axs[0].plot(time, xx_real_mpc[0,:Tsim],'m', linewidth=2, label='MPC')
axs[0].plot(time, xx_star[0,:Tsim],'--g', linewidth=2, label='Optimal')
axs[0].grid()
axs[0].set_ylabel('$V$')
axs[0].set_xlim([-1,Tsim])

#####
axs[1].plot(time, xx_real_mpc[1,:Tsim],'m', linewidth=2, label='MPC')
axs[1].plot(time, xx_star[1,:Tsim], '--g', linewidth=2, label='Optimal')
axs[1].grid()
axs[1].set_ylabel('$alpha$')
axs[1].set_xlim([-1,Tsim])

#####
axs[2].plot(time, xx_real_mpc[2,:Tsim],'m', linewidth=2, label='MPC')
axs[2].plot(time, xx_star[2,:Tsim], '--g', linewidth=2, label='Optimal')
axs[2].grid()
axs[2].set_ylabel('$theta$')
axs[2].set_xlim([-1,Tsim])

#####
axs[3].plot(time, xx_real_mpc[3,:Tsim],'m', linewidth=2, label='MPC')
axs[3].plot(time, xx_star[3,:Tsim], '--g', linewidth=2, label='Optimal')
axs[3].grid()
axs[3].set_ylabel('$q$')
axs[3].set_xlim([-1,Tsim])

#####
axs[4].plot(time, uu_real_mpc[0,:Tsim],'m', linewidth=2, label='MPC')
axs[4].plot(time, uu_star[0,:Tsim],'--g', linewidth=2, label='Optimal')
axs[4].grid()
axs[4].set_ylabel('$delta_t$')
axs[4].set_xlim([-1,Tsim-T_pred])

#####
axs[5].plot(time, uu_real_mpc[1,:Tsim],'m', linewidth=2, label='MPC')
axs[5].plot(time, uu_star[1,:Tsim],'--g', linewidth=2, label='Optimal')
axs[5].grid()
axs[5].set_ylabel('$delta_c$')
axs[5].set_xlim([-1,Tsim])

#####
axs[6].plot(time, uu_real_mpc[2,:Tsim],'m', linewidth=2, label='MPC')
axs[6].plot(time, uu_star[2,:Tsim],'--g', linewidth=2, label='Optimal')
axs[6].grid()
axs[6].set_ylabel('$delta_e$')
axs[6].set_xlabel('time')
axs[6].set_xlim([-1,Tsim])

fig.align_ylabels(axs)

plt.legend()
plt.show()

# Plotting the trajectory
print("Task 4 completed")
# plt.plot(xx_real_mpc[0,:]*np.cos(xx_real_mpc[2,:]-xx_real_mpc[1,:]), xx_real_mpc[0,:]*np.sin(xx_real_mpc[2,:]-xx_real_mpc[1,:]), label='MPC Trajectory')
# plt.plot(xx_star[0,:]*np.cos(xx_star[2,:]-xx_star[1,:]), xx_star[0,:]*np.sin(xx_star[2,:]-xx_star[1,:]),'m--',label='Optimal Trajectory')
# plt.title('Airplane Trajectory')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()

# Define time interval
delta_t = param.dt  # for example 0.1 seconds

# Get the velocities in x and y
vx_star = xx_star[0, :] * np.cos(-xx_star[2, :] + xx_star[1, :])
vy_star = xx_star[0, :] * np.sin(xx_star[1, :] - xx_star[2, :])
vx_ref = xx_real_mpc[0, :] * np.cos(-xx_real_mpc[2, :] + xx_real_mpc[1, :])
vy_ref = xx_real_mpc[0, :] * np.sin(xx_real_mpc[1, :] - xx_real_mpc[2, :])

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
plt.plot(x_star, y_star, label='Optimal Trajectory')
plt.plot(x_ref, y_ref, 'm--', label='Reference Trajectory')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.title('Airplane Trajectories')
plt.show()


