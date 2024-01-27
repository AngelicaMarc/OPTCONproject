# OPTCONproject
This project deals with the design and implementation of an optimal control law for a supersonic aircraft with nonlinear drag and lift

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