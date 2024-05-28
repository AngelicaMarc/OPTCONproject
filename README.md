# OPTCONproject
This project deals with the design and implementation of an optimal control law for a supersonic aircraft with nonlinear drag and lift

# TODO

#vector.copy() non ce n'è
Errors in Jacobian
Armijo Debug
MPC: last T_pred instants: extend onto the second equilibria
Weight in cost matrices
Fix animation
Report



 save (ctrl+s)
 git add .
 git commit -m "commit"
 git push

 matrix multiplication
 for t in range(T-1):

    xx[t+1] = A@xx[t] + B@uu[t]
    yy[t] = C@xx[t] + D@uu[t]

For the equilibria: put V fixed with respect to the Mach number, alfa = 0, theta has to compensate for dynamics in alfa dot, else it will not go to zero

uu[0] = 0.0  # Throttle
uu[2] = ((((2*mm*gg*np.cos(theta))/(rho*VV**2))-CL-BB[0,0]*CM)/BB[0,1])/(1- (BB[1,1]*BB[0,0])/(BB[1,0]*BB[0,1]))  # Elevator
uu[1] = (CM - BB[1,1]*uu[2])/BB[1,0]  # Canard

abc = -0.5 * rho * VV**2 * (CT*uu[0]-CD) / mm*gg
print(abc)
theta = np.arcsin(abc)  # pitch

MODEL PARAMETERS AND INITIALIZATION

We've defined all the model parameters, set initial conditions and initialized the state stack and the control input stack.

COMPUTATION OF DYNAMICS OF THE SYSTEM

We've first defined a conditional variable "flag" that return the continous-time derivatives if flag==False and the discrete-time dynamics when flag==True. The discretization is computed by the implementation of the forward Euler method, updating the state variables xxp at the next time step. 

COMPUTATION OF THE JACOBIAN

We've computed the jacobian matrix for the system dynamics, denoting with fx the partial derivatives of the dynamics with respect to the state variables (VV, alfa, theta, qq) and with fu the partial derivatives of the dynamics with respect to the inputs (TT, CC, EE) .

USAGE OF FUNCTION "func"

We use this function to update the state based on the control input.

DEFINITION OF MATRICES Qt and Rt

We've defined the matrices for linear quadratic optimal problem for the state stack and the input stack. We want high precision definition in state variables VV and alfa, so we use high numbers in matrices defined. (wrong)

COMPUTATION OF STAGE COST AND TERMINAL COST

We've computed the stage cost function and the terminal cost function. The first one compute the cost function until T-1 instant, instead the other one computes the terminal cost at T instant. These functions take values in input from the "parameters" and "cost" functions. In the first one are defined states and inputs, in the second one we've defined the reference trajectories.

GENERATION OF REFERENCE TRAJECTORY

We've generated reference trajectories for the state stack and the input stack

TASK 1

In Task 1, starting from two desired states we defined a function to obtain two input equilibria for our system and compute the reference trajectories for states and inputs.

Our goal is to obtain a variation from one equilibria to the other one for the states of velocity and the pitch, keeping alfa and pitch rate constant.
 
The next step was to compute the optimal trajectory to move from equilibrium xx1 to xx2 exploiting the regularized version of Newton's-like algorithm for optimal control. 
In order to find a trajectory to move from one equilibrium to the other, we started from a reference curve exploiting what we did before (a step function).
The code employs a Newton-like optimization algorithm, to compute the optimal transition trajectory between two equilibria. The optimization aims to minimize a cost function, denoted as J (discussed later), while ensuring smooth and stable trajectory transitions.
Newton's method is iteratively applied to minimize the cost function J by updating the control inputs (uu) to steer the system from one equilibrium to another. 
The optimization process begins with initial guesses for the control inputs (
u) and system states based on a reference trajectory (traj\_ref) defined earlier in the code.

The algorithm iterates over a predefined number of iterations, in our case 50 were enough to meet our standards. 
At each iteration, Newton's method computes a descent direction by solving a system of linear equations derived from the gradient and Hessian of the cost function J.

Using the computed descent direction, the control inputs (u) are updated to move the system towards the desired trajectory, minimizing the cost function and improving alignment with the reference trajectory.

The computed trajectory minimizes the cost function while ensuring smooth and stable transitions between equilibria, meeting the requirements of the task.

