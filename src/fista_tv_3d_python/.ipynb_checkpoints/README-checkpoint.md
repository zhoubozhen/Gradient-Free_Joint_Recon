# FISTA-TV_3D_Python

**Reference**

  * Beck A. and Teboulle M., “Fast gradient-based algorithms for constrained total variation
    image denoising and deblurring problems,” IEEE Trans. Image Process., vol. 18, no. 11,
    pp. 2419--2434 (2009) DOI: 10.1109/TIP.2009.2028250

  * Beck A. and Teboulle M., “A fast iterative shrinkage-thresholding algorithm for linear 
    inverse problems,” SIAM J. Imaging Sci., vol. 2, no. 1, pp. 183--202 (2009) DOI:
    10.1137/080716542

**Note**

  * Current `forward_prop.py` and `backward_prop.py` are for use of k-Wave, and they are incomplete.

  * The forward (line 61) and backward operation (line 76) depending on the user's choice (e.g. 
    interpolation model, k-Wave) can be plugged in.

  * `kgrid`, `medium`, `sensor`, and `input_args` are for use of k-Wave

