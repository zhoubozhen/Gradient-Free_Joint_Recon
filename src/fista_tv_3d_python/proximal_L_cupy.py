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
import numpy as np
import cupy as cp

# LINEAR OPERATOR
# def operator_L(X, Y, Z, Nx, Ny, Nz):
#   Xpq = cp.zeros((Nx + 1, Ny, Nz))
#   Ypq = cp.zeros((Nx, Ny + 1, Nz))
#   Zpq = cp.zeros((Nx, Ny, Nz + 1))
#   Xpq[1:Nx, :, :] = X
#   Ypq[:, 1:Ny, :] = Y
#   Zpq[:, :, 1:Nz] = Z
  
#   L = Xpq[1:Nx + 1, :, :] - Xpq[0:Nx, :, :] + \
#       Ypq[:, 1:Ny + 1, :] - Ypq[:, 0:Ny, :] + \
#       Zpq[:, :, 1:Nz + 1] - Zpq[:, :, 0:Nz]

#   return L

def operator_L(X, Y, Z, Nx, Ny, Nz):
  L = cp.zeros((Nx, Ny, Nz), dtype=cp.float32)
  L[:Nx-1,:,:] += X[:,:,:]
  L[1:,:,:] -= X[:,:,:]
  L[:,:Ny-1,:] += Y[:,:,:]
  L[:,1:,:] -= Y[:,:,:]
  L[:,:,:Nz-1] += Z[:,:,:]
  L[:,:,1:] -= Z[:,:,:]
  return L

# ADJOINT OF LINEAR OPERATOR
def operator_LT(p0, Nx, Ny, Nz):
  X = p0[0:Nx - 1, :, :] - p0[1:Nx, :, :]
  Y = p0[:, 0:Ny - 1, :] - p0[:, 1:Ny, :]
  Z = p0[:, :, 0:Nz - 1] - p0[:, :, 1:Nz]
  return X, Y, Z

# ORTHOGONAL PROJECTION OPERATOR ON THE SET C
def projection_C(p0, positive_constraint):
  p0_p = p0
  if positive_constraint:
    p0_p[p0 < 0] = 0
  return p0_p

# PROJECTION ONTO THE SET P
def projection_PX(X, Y, Z, Nx, Ny, Nz):
  denominator_X = cp.zeros((Nx - 1, Ny, Nz), dtype=cp.float32)

  denominator = cp.sqrt( \
    X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
    Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
    Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2)

  denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
  denominator_X[:, -1, :] = cp.abs(X[:, -1 ,:])
  denominator_X[:, :, -1] = cp.abs(X[:, : ,-1])
  denominator_X[denominator_X < 1] = 1

  X /= denominator_X

  return X

# def projection_PX(X, Y, Z, Nx, Ny, Nz):
#   denominator_X = cp.zeros((Nx - 1, Ny, Nz), dtype=cp.float32)
#   denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = cp.sqrt(denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1])
#   denominator_X[:, -1, :] = cp.abs(X[:, -1 ,:])
#   denominator_X[:, :, -1] = cp.abs(X[:, : ,-1])
#   denominator_X[denominator_X < 1] = 1
#   X /= denominator_X
#   return X

def projection_PY(X, Y, Z, Nx, Ny, Nz):
  denominator_Y = cp.zeros((Nx, Ny - 1, Nz), dtype=cp.float32)

  denominator = cp.sqrt( \
    X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
    Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
    Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2)
  
  denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
  denominator_Y[-1, :, :] = cp.abs(Y[-1, : ,:])
  denominator_Y[:, :, -1] = cp.abs(Y[:, : ,-1])
  denominator_Y[denominator_Y < 1] = 1

  Y /= denominator_Y

  return Y

# def projection_PY(X, Y, Z, Nx, Ny, Nz):
#   denominator_Y = cp.zeros((Nx, Ny - 1, Nz), dtype=cp.float32)
#   denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = cp.sqrt(denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1])
#   denominator_Y[-1, :, :] = cp.abs(Y[-1, : ,:])
#   denominator_Y[:, :, -1] = cp.abs(Y[:, : ,-1])
#   denominator_Y[denominator_Y < 1] = 1
#   Y /= denominator_Y
#   return Y

def projection_PZ(X, Y, Z, Nx, Ny, Nz):
  denominator_Z = cp.zeros((Nx, Ny, Nz - 1), dtype=cp.float32)

  denominator = cp.sqrt( \
    X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
    Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
    Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2)
  
  denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
  denominator_Z[-1, :, :] = cp.abs(Z[-1, : ,:])
  denominator_Z[:, -1, :] = cp.abs(Z[:, -1 ,:])
  denominator_Z[denominator_Z < 1] = 1

  Z /= denominator_Z

  return Z

# def projection_PZ(X, Y, Z, Nx, Ny, Nz):
#   denominator_Z = cp.zeros((Nx, Ny, Nz - 1), dtype=cp.float32)
#   denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] += Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2
#   denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = cp.sqrt(denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1])
#   denominator_Z[-1, :, :] = cp.abs(Z[-1, : ,:])
#   denominator_Z[:, -1, :] = cp.abs(Z[:, -1 ,:])
#   denominator_Z[denominator_Z < 1] = 1
#   Z /= denominator_Z
#   return Z

def projection_P(X, Y, Z, Nx, Ny, Nz):
  X1 = projection_PX(X, Y, Z, Nx, Ny, Nz)
  Y1 = projection_PY(X, Y, Z, Nx, Ny, Nz)
  Z1 = projection_PZ(X, Y, Z, Nx, Ny, Nz)

  return X1, Y1, Z1


# def projection_P(X, Y, Z, Nx, Ny, Nz):
#   denominator_X = cp.zeros((Nx - 1, Ny, Nz))
#   denominator_Y = cp.zeros((Nx, Ny - 1, Nz))
#   denominator_Z = cp.zeros((Nx, Ny, Nz - 1))

#   denominator = cp.sqrt( \
#     X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
#     Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
#     Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2)

#   denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
#   denominator_X[:, -1, :] = cp.abs(X[:, -1 ,:])
#   denominator_X[:, :, -1] = cp.abs(X[:, : ,-1])
#   denominator_X[denominator_X < 1] = 1
  
#   denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
#   denominator_Y[-1, :, :] = cp.abs(Y[-1, : ,:])
#   denominator_Y[:, :, -1] = cp.abs(Y[:, : ,-1])
#   denominator_Y[denominator_Y < 1] = 1
  
#   denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
#   denominator_Z[-1, :, :] = cp.abs(Z[-1, : ,:])
#   denominator_Z[:, -1, :] = cp.abs(Z[:, -1 ,:])
#   denominator_Z[denominator_Z < 1] = 1

#   X = X/denominator_X
#   Y = Y/denominator_Y
#   Z = Z/denominator_Z

#   return X, Y, Z

# PROXIMAL MAP
# def proximal_L(p0, Nx, Ny, Nz, reg_param, positive_constraint, Niter):
#   X = cp.zeros((Nx - 1, Ny, Nz), dtype=np.float32) #Kevin
#   Y = cp.zeros((Nx, Ny - 1, Nz), dtype=np.float32) #Kevin
#   Z = cp.zeros((Nx, Ny, Nz - 1), dtype=np.float32) #Kevin
#   Xrs = X
#   Yrs = Y
#   Zrs = Z
#   Xpqp = X
#   Ypqp = Y
#   Zpqp = Z

#   tp = 1
#   p0_cp = cp.array(p0, dtype=cp.float32)
#   if reg_param!=0:
#     for iter in range(Niter):
#       L = operator_L(Xrs, Yrs, Zrs, Nx, Ny, Nz)
#       C = projection_C(p0_cp - reg_param*L, positive_constraint)
#       XLT, YLT, ZLT = operator_LT(C, Nx, Ny, Nz)
#       Xpqn, Ypqn, Zpqn = projection_P(Xrs + XLT/(8*reg_param), \
#                                       Yrs + YLT/(8*reg_param), \
#                                       Zrs + ZLT/(8*reg_param), \
#                                       Nx, Ny, Nz)
#       tn = (1 + cp.sqrt(1 + 4*tp**2))/2
#       Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
#       Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)
#       Zrs = Zpqp + ((tp - 1)/tn)*(Zpqn - Zpqp)

#       Xpqp = Xpqn
#       Ypqp = Ypqn
#       Zpqp = Zpqn
#       tp = tn

#     L = operator_L(Xpqn, Ypqn, Zpqn, Nx, Ny, Nz)
#     p0est = projection_C(p0_cp - reg_param*L, positive_constraint)
#   else:
#     for iter in range(Niter):
#       L = operator_L(Xrs, Yrs, Zrs, Nx, Ny, Nz)
#       C = projection_C(p0_cp, positive_constraint)
#       XLT, YLT, ZLT = operator_LT(C, Nx, Ny, Nz)
#       Xpqn, Ypqn, Zpqn = projection_P(Xrs, Yrs, Zrs,\
#                                       Nx, Ny, Nz)
#       tn = (1 + cp.sqrt(1 + 4*tp**2))/2
#       Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
#       Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)
#       Zrs = Zpqp + ((tp - 1)/tn)*(Zpqn - Zpqp)

#       Xpqp = Xpqn
#       Ypqp = Ypqn
#       Zpqp = Zpqn
#       tp = tn

#     L = operator_L(Xpqn, Ypqn, Zpqn, Nx, Ny, Nz)
#     p0est = projection_C(p0_cp, positive_constraint)

#   return cp.asnumpy(p0est)

def proximal_L(p0, Nx, Ny, Nz, reg_param, positive_constraint, Niter):
  Xrs = cp.zeros((Nx - 1, Ny, Nz), dtype=cp.float32) #Kevin
  Yrs = cp.zeros((Nx, Ny - 1, Nz), dtype=cp.float32) #Kevin
  Zrs = cp.zeros((Nx, Ny, Nz - 1), dtype=cp.float32) #Kevin
  Xpqp = Xrs
  Ypqp = Yrs
  Zpqp = Zrs

  tp = 1
  p0_cp = cp.array(p0, dtype=cp.float32)
  if reg_param!=0:
    for iter in range(Niter):
      L = operator_L(Xrs, Yrs, Zrs, Nx, Ny, Nz)
      C = projection_C(p0_cp - reg_param*L, positive_constraint)
      Xrs += C[0:Nx - 1, :, :]/(8*reg_param)
      Xrs -= C[1:Nx, :, :]/(8*reg_param)
      Yrs += C[:, 0:Ny - 1, :]/(8*reg_param)
      Yrs -= C[:, 1:Ny, :]/(8*reg_param)
      Zrs += C[:, :, 0:Nz - 1]/(8*reg_param)
      Zrs -= C[:, :, 1:Nz]/(8*reg_param)
      # Xpqn, Ypqn, Zpqn = projection_P(Xrs, Yrs, Zrs, \
      #                                 Nx, Ny, Nz)
      Xpqn = projection_PX(Xrs, Yrs, Zrs, Nx, Ny, Nz)
      Ypqn = projection_PY(Xrs, Yrs, Zrs, Nx, Ny, Nz)
      Zpqn = projection_PZ(Xrs, Yrs, Zrs, Nx, Ny, Nz)
      tn = (1 + cp.sqrt(1 + 4*tp**2))/2
      Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
      Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)
      Zrs = Zpqp + ((tp - 1)/tn)*(Zpqn - Zpqp)

      Xpqp = Xpqn
      Ypqp = Ypqn
      Zpqp = Zpqn
      tp = tn

    L = operator_L(Xpqn, Ypqn, Zpqn, Nx, Ny, Nz)
    p0est = projection_C(p0_cp - reg_param*L, positive_constraint)
  else:
    for iter in range(Niter):
      L = operator_L(Xrs, Yrs, Zrs, Nx, Ny, Nz)
      C = projection_C(p0_cp, positive_constraint)
      XLT, YLT, ZLT = operator_LT(C, Nx, Ny, Nz)
      Xpqn, Ypqn, Zpqn = projection_P(Xrs, Yrs, Zrs,\
                                      Nx, Ny, Nz)
      tn = (1 + cp.sqrt(1 + 4*tp**2))/2
      Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
      Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)
      Zrs = Zpqp + ((tp - 1)/tn)*(Zpqn - Zpqp)

      Xpqp = Xpqn
      Ypqp = Ypqn
      Zpqp = Zpqn
      tp = tn

    L = operator_L(Xpqn, Ypqn, Zpqn, Nx, Ny, Nz)
    p0est = projection_C(p0_cp, positive_constraint)

  return cp.asnumpy(p0est)