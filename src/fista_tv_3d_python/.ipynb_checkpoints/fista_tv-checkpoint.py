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
import time
# from forward_prop import forward_prop
# from backward_prop import backward_prop
from cost_func_tv import cost_func_tv
from proximal_L import *

# FISTA-TV
# def fista_tv(kgrid, medium, pmeas, sensor, input_args, \
#         reg_param, lip, positive_constraint, niter_max, cost_min, saving_dir): # for use of k-Wave
def fista_tv(pmeas, Nx, Ny, Nz, forward_op, adjoint_op, roi, \
        reg_param, lip, positive_constraint, niter_max, cost_min, saving_dir, \
             start_iter=1, init_guess=None, mpi_rank=0):

  if mpi_rank==0:
    print('FISTA-TV starts ...')

#   Nx = kgrid.Nx # Number of voxels in simulation domain along x-axis (for use of k-Wave)
#   Ny = kgrid.Ny # Number of voxels in simulation domain along y-axis (for use of k-Wave)
#   Nz = kgrid.Nz # Number of voxels in simulation domain along z-axis (for use of k-Wave)

  pest = np.zeros(pmeas.shape, dtype=np.float32)

  p0p = np.zeros((Nx, Ny, Nz), dtype=np.float32)
  p0y = np.zeros((Nx, Ny, Nz), dtype=np.float32)
  tp  = 1

  cost = np.zeros((niter_max, 1))

  for iter in range(start_iter, niter_max + 1):
    iter_start_time = time.time()
    if mpi_rank==0:
      print('[ Iteration ' + str(iter) + ' ]')

    if iter > 1:
      if mpi_rank==0:
        print('    Forward propagarion...')
      ## REPLACE the following line with forward propagation
      pest = forward_op(p0y[roi>0])

    # Cost function
    if mpi_rank==0:
      print('    Cost function...')
    costn = cost_func_tv(pmeas, pest, reg_param, p0y, Nx, Ny, Nz)
    if mpi_rank==0:
      print('        cost_y: ' + '{:.4g}'.format(costn))
    cost[iter - 1] = costn
    fname = os.path.join(saving_dir, 'cost_function_lip_' + '{:.0e}'.format(lip) + '.mat')
    if mpi_rank==0:
      try:
        sio.savemat(fname, {'cost': cost})
      except:
        print('cannot save cost function, but keep iteration')

    # Backward propagation
    if mpi_rank==0:
      print('    Backward propagation...')
    pest = pest - pmeas
    ## REPLACE the following line with backward propagation
    dp0 = adjoint_op(pest).reshape((Nx, Ny, Nz))
    dp0[roi==0] = 0

    # Apply the TV proximal operator (Equations 3.13/3.14 in FISTA paper)
    if mpi_rank==0:
      print('    TV proximal operator...')
    p0 = proximal_L(p0y -  (2/lip)*dp0, Nx, Ny, Nz, 2*reg_param/lip, positive_constraint, 5)

    # Weight at next iteration  (Equation 3.15 in FISTA paper)
    tn = (1 + math.sqrt(1 + 4*tp**2))/2

    # Equation 3.16 in FISTA paper
    p0y = p0 + ((tp - 1)/tn)*(p0 - p0p)
    if mpi_rank==0:
      fname = os.path.join(saving_dir, 'p0_iter_' + str(iter) + '.DAT')
      try:
        p0.ravel().astype(np.float32).tofile(fname)
      except OSError as err:
        print(err)
        print("cannot save, but keep iteration")

    tp = tn
    p0p = p0
    iter_end_time = time.time()
    if mpi_rank==0:
      print(f'This iteration takes {iter_end_time-iter_start_time}')
      iter_start_time = iter_end_time