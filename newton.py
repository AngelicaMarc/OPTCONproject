import numpy as np
import matplotlib.pyplot as plt
import parameters as param
import cost as cst

#define params

ns = param.ns  # number of states
ni = param.ni  # number of inputs
dt = param.dt  # sample time

dt = param.dt           # discretization step from dynamics
ts = param.num_steps    # number of time steps

tf = ts * dt            # Final time in seconds
tm = int(ts / 2)        # Middle time step

term_cond = 1e-8        #terminal condition

# ARMIJO PARAMETERS
c = 0.5
beta = 0.7
armijo_maxiters = 20    # number of Armijo iterations
stepsize_0 = 1          # initial stepsize
armijo_plt = 0          # plot Armijo 

# Import the cost matrices from costs
Qt = cst.QQt
Rt = cst.RRt
QT = cst.QQT

def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, ts, x0, qqin = None, rrin = None, qqfin = None, ccin = None):
    
    try:
        # check if matrix is (.. x .. x ts) - 3 dimensional array 
        ns, lA = AAin.shape[1:]
    except:
        # if not 3 dimensional array, make it (.. x .. x 1)
        AAin = AAin[:,:,None]
        ns, lA = AAin.shape[1:]

    try:  
        ni, lB = BBin.shape[1:]
    except:
        BBin = BBin[:,:,None]
        ni, lB = BBin.shape[1:]

    try:
        nQ, lQ = QQin.shape[1:]
    except:
        QQin = QQin[:,:,None]
        nQ, lQ = QQin.shape[1:]

    try:
        nR, lR = RRin.shape[1:]
    except:
        RRin = RRin[:,:,None]
        nR, lR = RRin.shape[1:]

    try:
        nSi, nSs, lS = SSin.shape
    except:
        SSin = SSin[:,:,None]
        nSi, nSs, lS = SSin.shape

    # Check dimensions consistency -- safety
    if nQ != ns:
        print("Matrix Q does not match number of states")
        exit()
    if nR != ni:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != ns:
        print("Matrix S does not match number of states")
        exit()
    if nSi != ni:
        print("Matrix S does not match number of inputs")
        exit()


    if lA < ts:
        AAin = AAin.repeat(ts, axis=2)
    if lB < ts:
        BBin = BBin.repeat(ts, axis=2)
    if lQ < ts:
        QQin = QQin.repeat(ts, axis=2)
    if lR < ts:
        RRin = RRin.repeat(ts, axis=2)
    if lS < ts:
        SSin = SSin.repeat(ts, axis=2)

    # Check for affine terms


    KK = np.zeros((ni, ns, ts))
    sigma = np.zeros((ni, ts))
    PP = np.zeros((ns, ns, ts))
    pp = np.zeros((ns, ts))

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin
    
    qq = qqin
    rr = rrin

    qqf = qqfin

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, ts))
    uu = np.zeros((ni, ts))

    xx[:,0] = x0
    
    PP[:,:,-1] = QQf
    pp[:,-1] = qqf
    
    # Solve Riccati equation
    for tt in reversed(range(ts-1)):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:, tt+1][:,None]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        # print(MMt_inv)
        mmt = rrt + BBt.T @ pptp
        
        PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
        
        ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()


    # Evaluate KK
    
    for tt in range(ts-1):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]

        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None]

        # Check positive definiteness

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp

        # for other purposes we could add a regularization step here...

        KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
        sigma_t = -MMt_inv@mmt

        sigma[:,tt] = sigma_t.squeeze()


    for tt in range(ts - 1):
        
        # Trajectory
        uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
        xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

        xx[:,tt+1] = xx_p
        
    return xx, uu, KK, sigma

def Newton (xx, uu, xx_ref, uu_ref, x0, max_iters):

    # arrays to store data
    A = np.zeros((ns, ns, ts))
    B = np.zeros((ns, ni, ts))
    d1l = np.zeros((ns, ts))
    d2l = np.zeros((ni, ts))

    cc = np.zeros((ns,ts))
    xx0 = np.zeros((ns,))

    Qtilda = np.zeros((ns, ns, ts))
    Rtilda = np.zeros((ni, ni, ts))
    Stilda = np.zeros((ni, ns, ts))

    lmbd = np.zeros((ns, ts, max_iters+1))    # lambdas - costate seq.

    dJ = np.zeros((ni,ts, max_iters+1))       # DJ - gradient of J wrt u
    J = np.zeros(max_iters+1)                 # collect cost
    descent = np.zeros(max_iters+1)           # collect descent direction
    descent_arm = np.zeros(max_iters+1)       # collect descent direction     

    Dx = np.zeros((ns, ts, max_iters+1))
    Du = np.zeros((ni, ts, max_iters+1))

    ################################################################################################################

    for kk in range(max_iters):
        J[kk] = 0

        # Parameters evaluation

        for tt in range(ts):
            temp_cost= cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
            J[kk] += temp_cost
            fx, fu = param.jacobian(xx[:,tt,kk], uu[:,tt,kk])
            A[:,:,tt] = fx.T
            B[:,:,tt] = fu.T 

        
        temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
        J[kk] += temp_cost

        # Descent direction calculation
        lmbd_temp = cst.termcost(xx[:,ts-1,kk], xx_ref[:,ts-1])[1]
        lmbd[:,ts-1,kk] = lmbd_temp.squeeze()
        

        for tt in reversed(range(ts-1)):                              # integration backward in time

            a, b = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:3]          

            d1l[:,tt] = a.squeeze()
            d2l[:,tt] = b.squeeze()

            lmbd_temp = A[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + a       # costate equation
            dJ_temp = B[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + b         # gradient of J wrt u 
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()


        # Matrices evaluation
        for tt in range(ts):
            Qtilda[:,:,tt], Rtilda[:,:,tt], Stilda[:,:,tt] = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[3:]
        
        d1lT, QTilda = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[1:3]
        Dx[:,:,kk], Du[:,:,kk], KK, sigma = ltv_LQR(A, B, Qtilda, Rtilda, Stilda, QTilda, ts, xx0, d1l, d2l, d1lT.squeeze(), cc)


        for tt in reversed(range(ts)): 
            descent[kk] += Du[:,tt,kk].T@Du[:,tt,kk] 
            descent_arm[kk] += dJ[:,tt,kk].T@Du[:,tt,kk] 

        # Stepsize selection - ARMIJO
        stepsizes = []  # list of stepsizes
        costs_armijo = []

        stepsize = stepsize_0

        if 1:           # to change if you want to use costant stepsize (you also need to change stepsize_0 at the beginning)
            for ii in range(armijo_maxiters):

                # temp solution update

                xx_temp = np.zeros((ns,ts))
                uu_temp = np.zeros((ni,ts))

                xx_temp[:,0] = x0

                for tt in range(ts-1):
                    uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + stepsize*sigma[:,tt]
                    xx_temp[:,tt+1] = param.dynamics(xx_temp[:,tt], uu_temp[:,tt])

                JJ_temp = 0

                for tt in range(ts):
                    temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost

                temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                JJ_temp += temp_cost

                stepsizes.append(stepsize)                              # save the stepsize
                costs_armijo.append(np.min([JJ_temp, 100*J[kk]]))       # save the cost associated to the stepsize

                if JJ_temp > J[kk] + c*stepsize*descent_arm[kk]:
                    # update the stepsize
                    stepsize = beta*stepsize

                else:
                    print('Armijo stepsize = {:.3e}'.format(stepsize))
                    break
        
        # Armijo plot
        if armijo_plt:
            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):

                step = steps[ii]
                # temp solution update

                xx_temp = np.zeros((ns,ts))
                uu_temp = np.zeros((ni,ts))

                xx_temp[:,0] = x0

                for tt in range(ts-1):
                    uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + step*sigma[:,tt]
                    xx_temp[:,tt+1] = param.dynamics(xx_temp[:,tt], uu_temp[:,tt])

                # temp cost calculation
                JJ_temp = 0

                for tt in range(ts):
                    temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost

                temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                JJ_temp += temp_cost

                costs[ii] = np.min([JJ_temp, 100*J[kk]])

            plt.figure(1)
            plt.clf()
            plt.plot(steps, costs, color='g', label='$\\ell(x^k - \\gamma*d^k$)')
            plt.plot(steps, J[kk] + descent_arm[kk]*steps, color='r', label='$\\ell(x^k) - \\gamma*\\nabla\\ell(x^k)^{\\top}d^k$')
            plt.plot(steps, J[kk] + c*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$\\ell(x^k) - \\gamma*c*\\nabla\\ell(x^k)^{\\top}d^k$')
            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()
            plt.show()

        # Update the current solution

        xx_temp = np.zeros((ns,ts))
        uu_temp = np.zeros((ni,ts))

        xx_temp[:,0] = x0

        for tt in range(ts-1):
            uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + stepsize*sigma[:,tt]
            xx_temp[:,tt+1] = param.dynamics(xx_temp[:,tt], uu_temp[:,tt])


        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Termination condition
    
        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk+1,descent[kk], J[kk]))

        if descent[kk] <= term_cond:
            break

    return xx, uu, descent, J, kk


