# OPTCONproject
This project deals with the design and implementation of an optimal control law for a supersonic aircraft with nonlinear drag and lift

# TODO

Gradient to 1e-6 
vector.copy()
Errors in Jacobian
Armijo Debug
MPC: last T_pred instants: extend onto the second equilibria
Plots: INTEGRATE velocity (forward euler?)
Weight in cost matrices
Fix animation
Perturb initial condition
Report
Plots


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

