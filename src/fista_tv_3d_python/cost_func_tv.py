'''
author  - Seonyeong Park
date    - Dec 17, 2021

COST FUNCTION

Reference: 

* Beck A. and Teboulle M., “Fast gradient-based algorithms for constrained total variation
    image denoising and deblurring problems,” IEEE Trans. Image Process., vol. 18, no. 11,
    pp. 2419--2434 (2009) DOI: 10.1109/TIP.2009.2028250

* Beck A. and Teboulle M., “A fast iterative shrinkage-thresholding algorithm for linear 
    inverse problems,” SIAM J. Imaging Sci., vol. 2, no. 1, pp. 183--202 (2009) DOI:
    10.1137/080716542
'''

import numpy as np

# COST FUNCTION
def cost_func_tv(pmeas, pest, reg_param, p0, Nx, Ny, Nz):
  if reg_param != 0:
    # Compute TV norm of A (p0)
    dx = np.zeros((p0.shape), dtype=np.float32) #Kevin
    dy = np.zeros((p0.shape), dtype=np.float32) #Kevin
    dz = np.zeros((p0.shape), dtype=np.float32) #Kevin
    dx[0:Nx - 1, :, :] = p0[0:Nx - 1, :, :] - p0[1:Nx, :, :]
    dy[:, 0:Ny - 1, :] = p0[:, 0:Ny - 1, :] - p0[:, 1:Ny, :]
    dz[:, :, 0:Nz - 1] = p0[:, :, 0:Nz - 1] - p0[:, :, 1:Nz]
    tv = (np.sqrt(dx**2 + dy**2 + dz**2))
    tv_norm = np.sum(tv.ravel())
    
    cost = np.linalg.norm(pmeas - pest)**2 + 2*reg_param*tv_norm
  else:
    cost = np.linalg.norm(pmeas - pest)**2

  return cost