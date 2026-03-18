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
def operator_L(X, Y, Z, Nx, Ny, Nz):
    L = np.zeros((Nx, Ny, Nz), dtype=np.float32)
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

def projection_denominator(X, Y, Z, Nx, Ny, Nz):
    X_gpu = cp.array(X,dtype=cp.float32)
    Y_gpu = cp.array(Y,dtype=cp.float32)
    Z_gpu = cp.array(Z,dtype=cp.float32)
    denominator = cp.sqrt( \
        X_gpu[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
        Y_gpu[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2 + \
        Z_gpu[0:Nx - 1, 0:Ny - 1, 0:Nz - 1]**2)
    del X_gpu, Y_gpu, Z_gpu
    return denominator

def projection_X(X, denominator, Nx, Ny, Nz):
    X_gpu = cp.array(X,dtype=cp.float32)
    denominator_X = cp.zeros((Nx - 1, Ny, Nz),dtype=cp.float32)
    denominator_X[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
    denominator_X[:, -1, :] = cp.abs(X_gpu[:, -1 ,:])
    denominator_X[:, :, -1] = cp.abs(X_gpu[:, : ,-1])
    denominator_X[denominator_X < 1] = 1
    X_gpu = X_gpu/denominator_X
    X = cp.asnumpy(X_gpu)
    del X_gpu
    return X

def projection_Y(Y, denominator, Nx, Ny, Nz):
    Y_gpu = cp.array(Y,dtype=cp.float32)
    denominator_Y = cp.zeros((Nx, Ny - 1, Nz),dtype=cp.float32)
    denominator_Y[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
    denominator_Y[-1, :, :] = cp.abs(Y_gpu[-1, : ,:])
    denominator_Y[:, :, -1] = cp.abs(Y_gpu[:, : ,-1])
    denominator_Y[denominator_Y < 1] = 1
    Y_gpu = Y_gpu/denominator_Y
    Y = cp.asnumpy(Y_gpu)
    del Y_gpu
    return Y

def projection_Z(Z, denominator, Nx, Ny, Nz):
    Z_gpu = cp.array(Z,dtype=cp.float32)
    denominator_Z = cp.zeros((Nx, Ny, Nz - 1),dtype=cp.float32)
    denominator_Z[0:Nx - 1, 0:Ny - 1, 0:Nz - 1] = denominator
    denominator_Z[-1, :, :] = cp.abs(Z_gpu[-1, : ,:])
    denominator_Z[:, -1, :] = cp.abs(Z_gpu[:, -1 ,:])
    denominator_Z[denominator_Z < 1] = 1
    Z_gpu = Z_gpu/denominator_Z
    Z = cp.asnumpy(Z_gpu)
    del Z_gpu
    return Z

def projection_P(X, Y, Z, Nx, Ny, Nz):
    denominator = projection_denominator(X, Y, Z, Nx, Ny, Nz)
    X = projection_X(X, denominator, Nx, Ny, Nz)
    Y = projection_Y(Y, denominator, Nx, Ny, Nz)
    Z = projection_Z(Z, denominator, Nx, Ny, Nz)
    del denominator
    return X,Y,Z

# PROXIMAL MAP
def proximal_L(p0, Nx, Ny, Nz, reg_param, positive_constraint, Niter):
    X = np.zeros((Nx - 1, Ny, Nz), dtype=np.float32) #Kevin
    Y = np.zeros((Nx, Ny - 1, Nz), dtype=np.float32) #Kevin
    Z = np.zeros((Nx, Ny, Nz - 1), dtype=np.float32) #Kevin
    Xrs = X
    Yrs = Y
    Zrs = Z
    Xpqp = X
    Ypqp = Y
    Zpqp = Z

    tp = 1
    if reg_param!=0:
        for iter in range(Niter):
            L = operator_L(Xrs, Yrs, Zrs, Nx, Ny, Nz)
            C = projection_C(p0 - reg_param*L, positive_constraint)
            XLT, YLT, ZLT = operator_LT(C, Nx, Ny, Nz)  # gradient
            Xpqn, Ypqn, Zpqn = projection_P(Xrs + XLT/(8*reg_param), \
                                            Yrs + YLT/(8*reg_param), \
                                            Zrs + ZLT/(8*reg_param), \
                                            Nx, Ny, Nz)
            tn = (1 + np.sqrt(1 + 4*tp**2))/2
            Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
            Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)
            Zrs = Zpqp + ((tp - 1)/tn)*(Zpqn - Zpqp)

            Xpqp = Xpqn
            Ypqp = Ypqn
            Zpqp = Zpqn
            tp = tn

        L = operator_L(Xpqn, Ypqn, Zpqn, Nx, Ny, Nz)
        p0est = projection_C(p0 - reg_param*L, positive_constraint)
    else:
        for iter in range(Niter):
            L = operator_L(Xrs, Yrs, Zrs, Nx, Ny, Nz)
            C = projection_C(p0, positive_constraint)
            XLT, YLT, ZLT = operator_LT(C, Nx, Ny, Nz)
            Xpqn, Ypqn, Zpqn = projection_P(Xrs, Yrs, Zrs,\
                                            Nx, Ny, Nz)
            tn = (1 + np.sqrt(1 + 4*tp**2))/2
            Xrs = Xpqp + ((tp - 1)/tn)*(Xpqn - Xpqp)
            Yrs = Ypqp + ((tp - 1)/tn)*(Ypqn - Ypqp)
            Zrs = Zpqp + ((tp - 1)/tn)*(Zpqn - Zpqp)

            Xpqp = Xpqn
            Ypqp = Ypqn
            Zpqp = Zpqn
            tp = tn

        L = operator_L(Xpqn, Ypqn, Zpqn, Nx, Ny, Nz)
        p0est = projection_C(p0, positive_constraint)

    return p0est