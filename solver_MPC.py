import cvxpy as cp

def linear_mpc(AA_opt, BB_opt, QQ, RR, QQf, xxt, xx_opt, uu_opt, T_pred):

  xxt = xxt.squeeze()

  ns, ni = BB_opt.shape[:2]

  xx_mpc = cp.Variable((ns, T_pred))
  uu_mpc = cp.Variable((ni, T_pred))

  cost = 0
  constr = []

  for tt in range(T_pred-1):
    cost += cp.quad_form(xx_mpc[:,tt] - xx_opt[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt] - uu_opt[:,tt], RR)
    constr += [xx_mpc[:,tt+1] - xx_opt[:,tt+1]== AA_opt[:,:,tt]@(xx_mpc[:,tt] - xx_opt[:,tt]) + BB_opt[:,:,tt]@(uu_mpc[:,tt]-uu_opt[:,tt])]

  cost += cp.quad_form(xx_mpc[:,T_pred-1] - xx_opt[:, T_pred-1], QQf)
  
  constr += [xx_mpc[:,0] == xxt]
  problem = cp.Problem(cp.Minimize(cost), constr)
  problem.solve()

  if problem.status == "infeasible":
  # Otherwise, problem.value is inf or -inf, respectively.
    print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

  return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value