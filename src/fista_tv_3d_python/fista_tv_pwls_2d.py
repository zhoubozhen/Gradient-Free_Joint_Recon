'''''
author  - Seonyeong Park
date    - Dec 17, 2021

FISTA-TV

Reference: 

  * Beck A. and Teboulle M., “Fast gradient-based algorithms for constrained total variation
    image denoising and deblurring problems,” IEEE Trans. Image Process., vol. 18, no. 11,
    pp. 2419--2434 (2009) DOI: 10.1109/TIP.2009.2028250

  * Beck A. and Teboulle M., “A fast iterative shrinkage-thresholding algorithm for linear 
    inverse problems,” SIAM J. Imaging Sci., vol. 2, no. 1, pp. 183--202 (2009) DOI:
    10.1137/080716542

Note:

  * The forward (line 61) and backward operation (line 76) depending on the user's choice (e.g. 
    interpolation model, k-Wave) can be plugged in.

  * `kgrid`, `medium`, `sensor`, and `input_args` are for use of k-Wave

'''

import os
import math
import numpy as np
import scipy.io as sio
from cost_func_tv_2d import cost_func_tv
from proximal_L_2d import *

# FISTA-TV
def fista_tv(pmeas, Nx, Ny, forward_op, adjoint_op, roi,\
        reg_param, lip, positive_constraint, niter_max, cost_min, saving_dir, weighted_matrix=None):

  print('FISTA-TV starts ...')

#   Nx = kgrid.Nx # Number of voxels in simulation domain along x-axis (for use of k-Wave)
#   Ny = kgrid.Ny # Number of voxels in simulation domain along y-axis (for use of k-Wave)
#   Nz = kgrid.Nz # Number of voxels in simulation domain along z-axis (for use of k-Wave)

  pest = np.zeros(pmeas.shape, dtype=np.float32)

  p0p = np.zeros((Nx, Ny), dtype=np.float32)
  p0y = np.zeros((Nx, Ny), dtype=np.float32)
  dp0 = np.zeros((Nx, Ny), dtype=np.float32)
  tp  = 1

  cost = np.zeros((niter_max, 1))
  if weighted_matrix is not None:
    wpmeas = pmeas.copy()
    wpmeas = wpmeas*weighted_matrix
    pmeas = wpmeas

  for iter in range(1, niter_max + 1):
    print('[ Iteration ' + str(iter) + ' ]')

    if iter > 1:
      print('    Forward propagarion...')
      ## REPLACE the following line with forward propagation
      pest = forward_op(p0y[roi>0])
      if weighted_matrix is not None:
        pest = pest*weighted_matrix

    # Cost function
    print('    Cost function...')
    costn = cost_func_tv(pmeas, pest, reg_param, p0y, Nx, Ny)
    print('        cost_y: ' + '{:.4g}'.format(costn))
    cost[iter - 1] = costn
    fname = os.path.join(saving_dir, 'cost_function_lip_' + '{:.0e}'.format(lip) + '.mat')
    sio.savemat(fname, {'cost': cost})

    # Backward propagation
    print('    Backward propagation...')
    pest = pest - pmeas
    ## REPLACE the following line with backward propagation
    dp0 = adjoint_op(pest)
    dp0[roi==0] = 0

    # Apply the TV proximal operator (Equations 3.13/3.14 in FISTA paper)
    print('    TV proximal operator...')
    p0 = proximal_L(p0y -  (2/lip)*dp0, Nx, Ny, 2*reg_param/lip, positive_constraint, 50)

    # Weight at next iteration  (Equation 3.15 in FISTA paper)
    tn = (1 + math.sqrt(1 + 4*tp**2))/2

    # Equation 3.16 in FISTA paper
    p0y = p0 + ((tp - 1)/tn)*(p0 - p0p)
    fname = os.path.join(saving_dir, 'p0_adjoint_iter_' + str(iter) + '.DAT')
    try:
      p0.ravel().astype(np.float32).tofile(fname)
    except OSError as err:
      print(err)
      print("cannot save, but keep iteration")

    tp = tn
    p0p = p0


