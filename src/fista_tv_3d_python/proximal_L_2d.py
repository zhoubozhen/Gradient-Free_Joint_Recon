'''
author  - Seonyeong Park
date    - Dec 17, 2021

PROXIMAL MAP

Reference: 

  * Beck A. and Teboulle M., “Fast gradient-based algorithms for constrained total variation
    image denoising and deblurring problems,” IEEE Trans. Image Process., vol. 18, no. 11,
    pp. 2419--2434 (2009) DOI: 10.1109/TIP.2009.2028250

  * Beck A. and Teboulle M., “A fast iterative shrinkage-thresholding algorithm for linear 
    inverse problems,” SIAM J. Imaging Sci., vol. 2, no. 1, pp. 183--202 (2009) DOI:
    10.1137/080716542
'''
import math
import numpy as np

# LINEAR OPERATOR
def operator_L(X, Y, Nx, Ny):
  Xpq = np.zeros((Nx + 1, Ny))
  Ypq = np.zeros((Nx, Ny + 1))
  Xpq[1:Nx, :] = X
  Ypq[:, 1:Ny] = Y
  
  L = Xpq[1:Nx + 1, :] - Xpq[0:Nx, :] + \
      Ypq[:, 1:Ny + 1] - Ypq[:, 0:Ny]
  return L

# ADJOINT OF LINEAR OPERATOR
def operator_LT(p0, Nx, Ny):
  X = p0[0:Nx - 1, :] - p0[1:Nx, :]
  Y = p0[:, 0:Ny - 1] - p0[:, 1:Ny]

  return X, Y

# ORTHOGONAL PROJECTION OPERATOR ON THE SET C
def projection_C(p0, positive_constraint):
  p0_p = p0
  if positive_constraint:
    p0_p[p0 < 0] = 0

  return p0_p

# PROJECTION ONTO THE SET P
def projection_P(X, Y, Nx, Ny):
  denominator_X = np.zeros((Nx - 1, Ny))
  denominator_Y = np.zeros((Nx, Ny - 1))

  denominator = np.sqrt( \
    X[0:Nx - 1, 0:Ny - 1]**2 + \
    Y[0:Nx - 1, 0:Ny - 1]**2)

  denominator_X[0:Nx - 1, 0:Ny - 1] = denominator
  denominator_X[:, -1] = np.abs(X[:, -1])
  denominator_X[denominator_X < 1] = 1
  
  denominator_Y[0:Nx - 1, 0:Ny - 1] = denominator
  denominator_Y[-1, :] = np.abs(Y[-1, :])
  denominator_Y[denominator_Y < 1] = 1
  
  X = X/denominator_X
  Y = Y/denominator_Y

  return X, Y

# PROXIMAL MAP
def proximal_L(p0, Nx, Ny, reg_param, positive_constraint, Niter):
  X = np.zeros((Nx - 1, Ny), dtype=np.float32)
  Y = np.zeros((Nx, Ny - 1), dtype=np.float32)
  Xrs = X
  Yrs = Y
  Xpqp = X
  Ypqp = Y

  tp = 1
  if reg_param!=0:
    for iter in range(Niter):
      L = operator_L(Xrs, Yrs, Nx, Ny)
      C = projection_C(p0 - reg_param*L, positive_constraint)
      XLT, YLT = operator_LT(C, Nx, Ny)
      Xpqn, Ypqn = projection_P(Xrs + XLT/(8*reg_param), \
                                      Yrs + YLT/(8*reg_param), \
                                      Nx, Ny)
      tn = (1 + math.sqrt(1 + 4*tp**2))/2
      Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
      Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)

      Xpqp = Xpqn
      Ypqp = Ypqn
      tp = tn

    L = operator_L(Xpqn, Ypqn, Nx, Ny)
    p0est = projection_C(p0 - reg_param*L, positive_constraint)
  else:
    for iter in range(Niter):
      L = operator_L(Xrs, Yrs, Nx, Ny)
      C = projection_C(p0, positive_constraint)
      XLT, YLT = operator_LT(C, Nx, Ny)
      Xpqn, Ypqn = projection_P(Xrs, Yrs, Nx, Ny)
      tn = (1 + math.sqrt(1 + 4*tp**2))/2
      Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
      Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)

      Xpqp = Xpqn
      Ypqp = Ypqn
      tp = tn

    L = operator_L(Xpqn, Ypqn, Nx, Ny)
    p0est = projection_C(p0, positive_constraint)
  return p0est