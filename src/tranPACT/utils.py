import numpy as np
import math
from scipy.ndimage import convolve

import warnings

__all__ = ['moczo_average','no_average', 'staggered_prop', 'aubry_method']

def sampler(b, kappa, h):
    '''
    generate texture signal by gauss spectral function
    more details in paper
    Franceschini E, Mensah S, Amy D, Lefebvre JP. A 2-D anatomic breast ductal computer phantom for ultrasonic imaging. IEEE Trans Ultrason Ferroelectr Freq Control. 2006;53(7):1281-1288. doi:10.1109/tuffc.2006.1665076
    b: input whitenoise
    kappa: correlation length [mm]
    h: pixel size[mm]
    '''
    bhat = np.fft.fftn(b)
    nx,ny,nz = b.shape
    Lx = nx*h
    Ly = ny*h
    Lz = nz*h
    kz = np.concatenate((np.linspace(0,int(nz/2-1),int(nz/2)),np.linspace(int(-nz/2),-1,int(nz/2))), axis=None)
    ky = np.concatenate((np.linspace(0,int(ny/2-1),int(ny/2)),np.linspace(int(-ny/2),-1,int(ny/2))), axis=None)
    kx = np.concatenate((np.linspace(0,int(nx/2-1),int(nx/2)),np.linspace(int(-nx/2),-1,int(nx/2))), axis=None)
    kz *= (2*math.pi/float(Lz))
    ky *= (2*math.pi/float(Ly))
    kx *= (2*math.pi/float(Lx))
    kkx, kky, kkz = np.meshgrid(ky,kx,kz)
    #d = np.power(((np.power(kkx,2) + np.power(kky,2)) + 1./(kappa*kappa)),alpha_over_2);
    d2 = np.exp(-kappa*kappa*(np.power(kkx,2) + np.power(kky,2) + np.power(kkz,2))/8)
    m = np.real(np.fft.ifftn(bhat*d2))
    return m

def no_average(r_dis, fn, water_index=2, texture_seed=None, **kwargs):
    if np.min(r_dis)==0:
        med_rho = np.ones(r_dis.shape, dtype='f')
        med_compress = np.ones(r_dis.shape, dtype='f')
        med_shear = np.ones(r_dis.shape, dtype='f')
        med_alpha = np.ones(r_dis.shape, dtype='f')

        for mid in np.arange(fn.shape[0]):
            med_compress[r_dis==mid] = np.sqrt((fn[mid,1]+4*fn[mid,2]/3)/fn[mid,0])
            med_shear[r_dis==mid] = np.sqrt(fn[mid,2]/fn[mid,0])
            med_rho[r_dis==mid] = fn[mid,0]  
            med_alpha[r_dis==mid] = fn[mid,3]
    else:
        med_compress = np.ones(r_dis.shape, dtype='f')
        med_shear = np.ones(r_dis.shape, dtype='f')
        med_rho = np.ones(r_dis.shape, dtype='f')
        med_alpha = np.ones(r_dis.shape, dtype='f')
        for mid in np.arange(-2,3):
            med_compress[r_dis==mid] = np.sqrt((fn[mid+water_index,1]+4*fn[mid+water_index,2]/3)/fn[mid+water_index,0])
            med_shear[r_dis==mid] = np.sqrt(fn[mid+water_index,2]/fn[mid+water_index,0])
            med_rho[r_dis==mid] = fn[mid+water_index,0]
            med_alpha[r_dis==mid] = fn[mid+water_index,3]
            if mid==-2 and (texture_seed is not None):
                np.random.seed(texture_seed)
                random_correlated = sampler(np.random.normal(0, 1, r_dis.shape), 4, 1)
                random_correlated /= np.std(random_correlated)
                random_correlated = random_correlated*0.073+0.73
                
                p_min, p_max = fn[water_index,0], fn[water_index+1,0]
                p_map = random_correlated*p_min + (1-random_correlated)*p_max       # density = porosity*d_water + (1-porosity)*d_bone
                med_rho[r_dis==mid] = p_map[r_dis==mid]
                p_min, p_max = np.sqrt((fn[water_index,1]+4*fn[water_index,2]/3)/fn[water_index,0]), np.sqrt((fn[water_index+1,1]+4*fn[water_index+1,2]/3)/fn[water_index+1,0])
                p_map = p_min + (1-random_correlated)*(p_max-p_min)                 # cl = cl_water + (1-porosity)*(cl_bone-cl_water)
                med_compress[r_dis==mid] = p_map[r_dis==mid]                        # compressional ws
                p_map /= 2                                                          # cs = cl/2
                med_shear[r_dis==mid] = p_map[r_dis==mid]                           # shear ws
                
    med_mu_diag = med_rho*med_shear**2
    med_lam_diag = med_rho*med_compress**2 - 2*med_mu_diag
    return (med_compress, med_shear, med_rho,
            med_alpha, med_mu_diag, med_lam_diag)

def aubry_method(ct_data,
                 den_bone=2520,cl_bone=4.0, cs_bone=2.0, alpha_bone=0.9,
                 den_water=1000, cl_water=1.48, cs_water=0, alpha_water=0.0,
                 cor_length=4, HU_based=1000, use_ct_smooth=True,
                 ct_smooth_path=None ):
    """
    Applies the Aubry method to compute material properties from CT data.

    This function calculates various material properties (compressional wave speed,
    shear wave speed, density, attenuation, Lamé parameters) based on input
    CT data and specified material properties for bone and water. It models
    the material as a mixture of bone and water based on a porosity selection
    derived from the CT values.

    Parameters
    ----------
    ct_data : numpy.ndarray
        Input CT (Computed Tomography) data. Expected to be a multi-dimensional
        array where higher values typically correspond to denser materials (e.g., bone).
        The data should be scaled such that `np.max(ct_data)` represents the
        maximum density (e.g., pure bone).
    den_bone : float, optional
        Density of bone in kg/m^3. Default is 2520.
    cl_bone : float, optional
        Compressional wave speed in bone in km/s. Default is 4.0.
    cs_bone : float, optional
        Share wave speed in bone in km/s. Default is 2.0.
    alpha_bone : float, optional
        Attenuation coefficient for bone. Default is 0.9.
    den_water : float, optional
        Density of water in kg/m^3. Default is 1000.
    cl_water : float, optional
        Compressional wave speed in water in km/s. Default is 1.48.
    cs_water : float, optional
        Shear wave speed in water in km/s. Default is 0.
    alpha_water : float, optional
        Attenuation coefficient for water. Default is 0.
    cor_length : int or None, optional
        Correlation length for the `sampler` function. If not None, the `ct_data`
        will be processed by `sampler` before property calculation.
        Default is 4.
    HU_based : float, optional
        The demoninator for the porosity calculation, typically 1000 for CT data.
        If the CT data is scaled for png already, this can be set to the None.
        This will lead to the use of the maximum value of the CT data.

    Returns
    -------
    tuple
        A tuple containing the calculated material properties:
        - med_compress : numpy.ndarray
            Calculated compressional wave speed (cl) for the mixed medium.
        - med_shear : numpy.ndarray
            Calculated shear wave speed (cs) for the mixed medium.
        - med_rho : numpy.ndarray
            Calculated density (rho) for the mixed medium.
        - med_alpha : numpy.ndarray
            Calculated attenuation coefficient (alpha) for the mixed medium.
        - med_mu_diag : numpy.ndarray
            Calculated Lamé's first parameter (mu) for the mixed medium.
        - med_lam_diag : numpy.ndarray
            Calculated Lamé's second parameter (lambda) for the mixed medium.

    Notes
    -----
    The "Aubry method" typically refers to models or approaches used in medical
    imaging or acoustics to derive material properties from CT scans, often
    in the context of bone mechanics or ultrasound propagation. The specific
    equations used here are based on a linear mixing model for density
    and compressional wave speed, and a power-law mixing model for attenuation.
    Shear wave speed and Lamé parameters are derived from these primary
    properties.

    Reference
    -----
    J.-F. Aubry, M. Tanter, M. Pernot, J.-L. Thomas, and M. Fink, “Experimental demonstration of noninvasive transskull adaptive focusing based on prior computed tomography scans,” The Journal of the Acoustical Society of America, vol. 113, no. 1, pp. 84–93, Jan. 2003, doi: 10.1121/1.1529663.


    Examples
    --------
    >>> import numpy as np
    >>> # Example CT data (e.g., a simple gradient from water to bone)
    >>> ct_example = np.linspace(0, 1000, 100).reshape(10,10)
    >>> cl, cs, rho, alpha, mu, lam = aubry_method(ct_example)
    >>> print(f"First element density: {rho.flatten()[0]:.2f} kg/m^3")
    First element density: 1000.00 kg/m^3
    >>> print(f"Last element density: {rho.flatten()[-1]:.2f} kg/m^3")
    Last element density: 2520.00 kg/m^3
    >>> # Example with sampler activated
    >>> cl_s, cs_s, rho_s, alpha_s, mu_s, lam_s = aubry_method(ct_example, cor_length=2)
    >>> print(f"First element density with sampling: {rho_s.flatten()[0]:.2f} kg/m^3")
    First element density with sampling: 1100.00 kg/m^3
    """
    ct_data_original = ct_data.copy()

    if use_ct_smooth:
        assert ct_smooth_path is not None
        ct_data = np.load(ct_smooth_path)["ct_smooth"]
        print("smooth ct data read successfully")
    else:
        if cor_length is not None:
            print("Computer sampler")
            ct_data = sampler(ct_data, cor_length, 1)


    # # Apply Gaussian spectral filtering to spetially correlated properties
    # ct_data_original = ct_data.copy()
    # if cor_length is not None:
    #     ct_data = sampler(ct_data, cor_length, 1)

    # Calculate the Phi (Eq. 3, porosity) in the Aubry's paper.
    HU_based = HU_based or np.max(ct_data)
    poro_select = 1 - ct_data / HU_based

    # Calculate mixed medium properties
    med_rho = poro_select * den_water + (1 - poro_select) * den_bone  # Density
    med_compress = cl_water + (1 - poro_select) * (cl_bone - cl_water) # Compressional wave speed
    med_shear = cs_water + (1 - poro_select) * (cs_bone - cs_water)  # Shear wave speed
    med_alpha = alpha_water + (alpha_bone - alpha_water) * (poro_select**0.5) # Attenuation
    med_alpha[ct_data_original==ct_data_original[1,1,1]] = alpha_water  # Ensure water region has correct attenuation

    # Calculate Lamé parameters
    med_mu_diag = med_rho * med_shear**2  # Shear modulus
    med_lam_diag = med_rho * med_compress**2 - 2 * med_mu_diag # First Lamé parameter

    del ct_data_original  # Clean up to free memory
    print("Aubry completed, enter moczo average")
    return (med_compress, med_shear, med_rho,
            med_alpha, med_mu_diag, med_lam_diag)

def moczo_average(med_rho, med_alpha, med_mu_origin, med_lam_origin):
    
    idx = np.zeros_like(med_rho)
    idx[1:-1,1:-1,1:-1] = 1
    
    def get_fil(x,y,z):
        if x>0 and y!=0 and z!=0:
            fil = np.array([[[1,1,0],[1,1,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        elif x<0 and y!=0 and z!=0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]]],dtype=np.float32)
        elif x>0 and y!=0 and z==0:
            fil = np.array([[[0,1,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        elif x<0 and y!=0 and z==0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]]],dtype=np.float32)
        elif x>0 and y==0 and z!=0:
            fil = np.array([[[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        elif x<0 and y==0 and z!=0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]]],dtype=np.float32)
        elif x==0 and y!=0 and z!=0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        elif x>0 and y==0 and z==0:
            fil = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        elif x<0 and y==0 and z==0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]]],dtype=np.float32)
        elif x==0 and y!=0 and z==0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        elif x==0 and y==0 and z!=0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        else:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
        return fil
    
    def a_smooth(x,fil):
        afterconvolve = convolve(x,fil, mode='constant')/np.sum(fil)
        return afterconvolve
    def h_smooth(x,fil):
        eps = 1e-6   # 或 1e-8，看你数据尺度
        x_safe = np.maximum(x, eps)

        afterconvolve = 1 / (
            convolve(1 / x_safe, fil, mode='constant') / np.sum(fil)
        )
        # afterconvolve = 1/(convolve(1/(x),fil, mode='constant')/np.sum(fil))
        return afterconvolve
    
    med_rho_x = a_smooth(med_rho,get_fil(-0.5,0.0,0.0))
    med_rho_x[idx==0] = med_rho[idx==0]
    med_rho_y = a_smooth(med_rho,get_fil(0.0,0.5,0.0))
    med_rho_y[idx==0] = med_rho[idx==0]
    med_rho_z = a_smooth(med_rho,get_fil(0.0,0.0,0.5))
    med_rho_z[idx==0] = med_rho[idx==0]
    
    med_alpha_x = a_smooth(med_alpha,get_fil(-0.5,0.0,0.0))
    med_alpha_x[idx==0] = med_alpha[idx==0]
    med_alpha_y = a_smooth(med_alpha,get_fil(0.0,0.5,0.0))
    med_alpha_y[idx==0] = med_alpha[idx==0]
    med_alpha_z = a_smooth(med_alpha,get_fil(0.0,0.0,0.5))
    med_alpha_z[idx==0] = med_alpha[idx==0]
    
    med_mu_diag = h_smooth(med_mu_origin,get_fil(0.0,0.0,0.0))
    med_mu_diag[idx==0] = med_mu_origin[idx==0]
    med_mu_xy = h_smooth(med_mu_origin,get_fil(-0.5,0.5,0.0))
    med_mu_xy[idx==0] = med_mu_origin[idx==0]
    med_mu_yz = h_smooth(med_mu_origin,get_fil(0.0,0.5,0.5))
    med_mu_yz[idx==0] = med_mu_origin[idx==0]
    med_mu_xz = h_smooth(med_mu_origin,get_fil(-0.5,0.0,0.5))
    med_mu_xz[idx==0] = med_mu_origin[idx==0]
    
    med_kappa_origin = med_lam_origin + (2*med_mu_origin)/3
    med_kappa_diag = h_smooth(med_kappa_origin,get_fil(0.0,0.0,0.0))
    med_kappa_diag[idx==0] = med_kappa_origin[idx==0]
    med_lam_diag = med_kappa_diag - (2*med_mu_diag)/3
    print("moczo average completed")
    return (med_rho_x, med_rho_y, med_rho_z, med_alpha_x, med_alpha_y, med_alpha_z,
            med_mu_diag, med_mu_xy, med_mu_yz, med_mu_xz, med_lam_diag)

def moczo_average_gpu(med_rho, med_alpha, med_mu_origin, med_lam_origin):
    """
    GPU version of moczo_average.
    - 完全对齐原版逻辑
    - a_smooth 用 GPU convolve
    - h_smooth 目前保持 return x（注释保留，方便你以后打开）
    - 输入/输出都是 numpy array（对上游无侵入）
    """

    import numpy as np
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cp_convolve

    # --------------------------------------------------
    # mask
    # --------------------------------------------------
    idx = np.zeros_like(med_rho, dtype=np.float32)
    idx[1:-1, 1:-1, 1:-1] = 1.0

    # --------------------------------------------------
    # filters（完全复制你的 if-else 结构）
    # --------------------------------------------------
    def get_fil(x, y, z):
        if x > 0 and y != 0 and z != 0:
            fil = np.array([[[1,1,0],[1,1,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        elif x < 0 and y != 0 and z != 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]]], np.float32)
        elif x > 0 and y != 0 and z == 0:
            fil = np.array([[[0,1,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        elif x < 0 and y != 0 and z == 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]]], np.float32)
        elif x > 0 and y == 0 and z != 0:
            fil = np.array([[[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        elif x < 0 and y == 0 and z != 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]]], np.float32)
        elif x == 0 and y != 0 and z != 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[1,1,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        elif x > 0 and y == 0 and z == 0:
            fil = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        elif x < 0 and y == 0 and z == 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]]], np.float32)
        elif x == 0 and y != 0 and z == 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,1,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        elif x == 0 and y == 0 and z != 0:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[1,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        else:
            fil = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                            [[0,0,0],[0,1,0],[0,0,0]],
                            [[0,0,0],[0,0,0],[0,0,0]]], np.float32)
        return fil

    # --------------------------------------------------
    # GPU helpers
    # --------------------------------------------------
    def a_smooth(x, fil):
        fil_gpu = cp.asarray(fil)
        return cp_convolve(x, fil_gpu, mode='constant') / cp.sum(fil_gpu)

    def h_smooth(x, fil):
        fil_gpu = cp.asarray(fil)

        invx = 1.0 / x
        tmp = cp_convolve(invx, fil_gpu, mode='constant') / cp.sum(fil_gpu)
        out = 1.0 / tmp

        # 显式释放中间变量（可选，但稳）
        del invx, tmp
        cp._default_memory_pool.free_all_blocks()
        print("h_smooth Additional variables deleted")
        return out



    # --------------------------------------------------
    # move to GPU
    # --------------------------------------------------
    rho  = cp.asarray(med_rho)
    alp  = cp.asarray(med_alpha)
    mu0  = cp.asarray(med_mu_origin)
    lam0 = cp.asarray(med_lam_origin)
    idxg = cp.asarray(idx)

    # --------------------------------------------------
    # rho
    # --------------------------------------------------
    rho_x = a_smooth(rho, get_fil(-0.5, 0.0, 0.0))
    rho_y = a_smooth(rho, get_fil( 0.0, 0.5, 0.0))
    rho_z = a_smooth(rho, get_fil( 0.0, 0.0, 0.5))
    rho_x = cp.where(idxg, rho_x, rho)
    rho_y = cp.where(idxg, rho_y, rho)
    rho_z = cp.where(idxg, rho_z, rho)

    # --------------------------------------------------
    # alpha
    # --------------------------------------------------
    alp_x = a_smooth(alp, get_fil(-0.5, 0.0, 0.0))
    alp_y = a_smooth(alp, get_fil( 0.0, 0.5, 0.0))
    alp_z = a_smooth(alp, get_fil( 0.0, 0.0, 0.5))
    alp_x = cp.where(idxg, alp_x, alp)
    alp_y = cp.where(idxg, alp_y, alp)
    alp_z = cp.where(idxg, alp_z, alp)

    # --------------------------------------------------
    # mu
    # --------------------------------------------------
    mu_diag = h_smooth(mu0, get_fil(0.0, 0.0, 0.0))
    mu_xy   = h_smooth(mu0, get_fil(-0.5, 0.5, 0.0))
    mu_yz   = h_smooth(mu0, get_fil( 0.0, 0.5, 0.5))
    mu_xz   = h_smooth(mu0, get_fil(-0.5, 0.0, 0.5))
    mu_diag = cp.where(idxg, mu_diag, mu0)
    mu_xy   = cp.where(idxg, mu_xy,   mu0)
    mu_yz   = cp.where(idxg, mu_yz,   mu0)
    mu_xz   = cp.where(idxg, mu_xz,   mu0)

    # --------------------------------------------------
    # lambda
    # --------------------------------------------------
    kappa0 = lam0 + (2.0 * mu0) / 3.0
    kappa_diag = h_smooth(kappa0, get_fil(0.0, 0.0, 0.0))
    kappa_diag = cp.where(idxg, kappa_diag, kappa0)
    lam_diag = kappa_diag - (2.0 * mu_diag) / 3.0

    print("moczo average (GPU) completed")

    # --------------------------------------------------
    # back to CPU
    # --------------------------------------------------
    return (
        cp.asnumpy(rho_x),
        cp.asnumpy(rho_y),
        cp.asnumpy(rho_z),
        cp.asnumpy(alp_x),
        cp.asnumpy(alp_y),
        cp.asnumpy(alp_z),
        cp.asnumpy(mu_diag),
        cp.asnumpy(mu_xy),
        cp.asnumpy(mu_yz),
        cp.asnumpy(mu_xz),
        cp.asnumpy(lam_diag),
    )


def staggered_prop(r_dis=None, fn=None, water_index=2, texture_seed=None, 
    medium_spatial=None, ct_data=None, cor_length=None,use_ct_smooth=True,
    ct_smooth_path=None):
    """
    Calculates material properties for a medium to be used in a staggered-grid
    numerical simulation, supporting different input modalities.

    This function serves as a flexible dispatcher for determining the material
    properties (density, attenuation, Lamé parameters, etc.) of a simulation
    medium. It supports three primary methods for property definition:
    1. Directly from pre-computed spatial medium properties (`medium_spatial`).
    2. Derived from CT scan data using the `aubry_method`, along with material
       properties for water and bone (`ct_data` and `fn`).
    3. Generated using a 'no average' method based on a radial distribution
       (`r_dis`) and material properties (`fn`), often for idealized or
       textured media.

    The final set of material properties is then passed through a `moczo_average`
    function for further processing or averaging suitable for staggered-grid
    simulations.

    Parameters
    ----------
    r_dis : numpy.ndarray, optional
        Radial distribution or spatial discretization data. Used when
        `medium_spatial` and `ct_data` are None, for the `no_average` method.
    fn : numpy.ndarray, optional
        A 2D array containing material properties for different components
        (e.g., water and bone). Expected to have 2 rows if `ct_data` is used:
        `fn[0, :]` for water properties and `fn[1, :]` for bone properties.
        The columns are assumed to represent `[density, bulk modulus, shear modulus, attenuation]`.
        Used with `ct_data` or `r_dis`.
    water_index : int, optional
        Index representing water properties within a larger structure (e.g., in `fn`).
        Used by the `no_average` method.
    texture_seed : int or None, optional
        Seed for texture generation. Used by the `no_average` method for
        reproducible textures. Default is None.
    medium_spatial : dict, optional
        A dictionary containing pre-computed spatially-varying material properties.
        Expected keys: 'rho' (density), 'alpha' (attenuation), 'cl' (compressional
        wave speed), 'cs' (shear wave speed). If provided, this dataset is used
        directly. Default is None.
    ct_data : numpy.ndarray, optional
        CT (Computed Tomography) scan data. If provided, the `aubry_method` is
        used to derive material properties based on `fn` for water and bone.
        Default is None.
    cor_length : int or None, optional
        Correlation length for spatial averaging or texture generation. If not None,
        it is used in the `aubry_method` to apply spatial correlation to the CT data

    Returns
    -------
    tuple
        A tuple containing the processed material properties for the medium,
        as returned by `moczo_average`.

    Raises
    ------
    ValueError
        If `ct_data` is provided but `fn` does not have exactly 2 rows, which
        are expected for water and bone properties.

    Notes
    -----
    The function prioritizes input methods in the following order:
    1. `medium_spatial` (highest priority)
    2. `ct_data` (second priority, requires `fn`)
    3. `r_dis` (lowest priority, requires `fn`, `water_index`, `texture_seed`)
    """
    # Initialize variables to hold derived properties
    med_rho, med_alpha, med_mu_origin, med_lam_origin = None, None, None, None

    if medium_spatial is not None:
        # Case 1: Directly use pre-computed spatial medium properties
        med_rho = medium_spatial['rho'].copy()
        med_alpha = medium_spatial['alpha'].copy()
        
        # Prioritize loading Lamé parameters (mu, lam) if both are explicitly provided.
        if 'mu' in medium_spatial and 'lam' in medium_spatial:
            med_mu_origin = medium_spatial['mu'].copy()
            med_lam_origin = medium_spatial['lam'].copy()

            # Issue a warning if 'cl' or 'cs' were also provided, indicating potential redundancy/inconsistency
            if 'cl' in medium_spatial or 'cs' in medium_spatial:
                warnings.warn(
                    "Both Lamé parameters ('mu', 'lam') AND wave speeds ('cl', 'cs') were found in 'medium_spatial'. "
                    "Lamé parameters were prioritized, and wave speeds were derived from them. "
                    "Please ensure consistency or provide only one set for elastic properties.",
                    UserWarning, stacklevel=2
                )

        # If mu and lam are NOT both provided, check if compressional (cl) and shear (cs) wave speeds are provided.
        # If cl and cs are available, derive mu and lam from them.
        elif 'cl' in medium_spatial and 'cs' in medium_spatial:
            med_cl = medium_spatial['cl'].copy()
            med_cs = medium_spatial['cs'].copy()

            # Derive Lamé parameters from wave speeds and density
            # Formulae: mu = rho * cs^2, lambda = rho * cl^2 - 2 * mu
            med_mu_origin = med_rho * med_cs**2
            med_lam_origin = med_rho * med_cl**2 - 2 * med_mu_origin

        else:
            # Error case: Neither a complete set of Lamé parameters nor wave speeds was provided.
            raise ValueError(
                "Insufficient elastic properties in 'medium_spatial'. "
                "Please provide either both 'cl' and 'cs' OR both 'mu' and 'lam', along with 'rho' and 'alpha'."
            )
    elif ct_data is not None:
        # Case 2: Derive properties from CT data using Aubry method
        if fn is None or fn.shape[0] != 2:
            raise ValueError("When 'ct_data' is provided, 'fn' must be a 2-row array "
                             "containing properties for water (row 0) and bone (row 1).")

        # Calculate compressional (cl) and shear (cs) wave speeds from fn
        # Assuming fn columns: [density, P-wave modulus (bulk+4mu/3), S-wave modulus (mu), alpha]
        cl = np.sqrt((fn[:, 1] + 4 * fn[:, 2] / 3) / fn[:, 0]) # cl = sqrt((lambda+2mu)/rho)
        cs = np.sqrt(fn[:, 2] / fn[:, 0]) # cs = sqrt(mu/rho)

        # Call aubry_method to get the mixed medium properties
        _, _, med_rho, med_alpha, med_mu_origin, med_lam_origin = aubry_method(
            ct_data,
            den_bone=fn[1, 0],
            cl_bone=cl[1],
            cs_bone=cs[1],
            alpha_bone=fn[1, 3],
            den_water=fn[0, 0],
            cl_water=cl[0],
            cs_water=cs[0],
            alpha_water=fn[0, 3],
            cor_length=cor_length,
            HU_based=None, use_ct_smooth=use_ct_smooth,
            ct_smooth_path=ct_smooth_path,
        )
    else:
        # Case 3: Derive properties using the 'no_average' method
        # This path is typically for idealized media or specific texture generation.
        if r_dis is None or fn is None:
            raise ValueError("When neither 'medium_spatial' nor 'ct_data' are provided, "
                             "'r_dis' and 'fn' must be specified for the 'no_average' method.")
        _, _, med_rho, med_alpha, med_mu_origin, med_lam_origin = no_average(
            r_dis, fn, water_index, texture_seed
        )

    # Apply Moczo averaging or final processing to the derived properties
    return moczo_average(med_rho, med_alpha, med_mu_origin, med_lam_origin)