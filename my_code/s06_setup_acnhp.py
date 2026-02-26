import numpy as np
import numpy.fft as fft
import math
import scipy.io as sio
import h5py

directory = '/shared/rsaas/hkhuang3/KH250319_headphantom/data/'
head_dir = '/shared/anastasio-s3/CommonData/bozhen/head/'
import argparse
import os

parser = argparse.ArgumentParser('Example usage: python3 s06_setup_acnhp.py -o 3')
parser.add_argument('-o','--opt', type=int, default=4,
                    help='Acoustic phantom option: 0 for 1p1l, 1 for 3p3l, 2 for diploe, 3 for CT-based, 4 for CT-based with smoothing')
parser.add_argument('-v','--ver', type=int, required=False, default=None,
                    help='Version of the acoustic phantom setup')
# parser.add_argument('--tr_prop', action='store_true',
#                     help='Enable nominal property phantom if this flag is present')
args = parser.parse_args()

#
skull_3p3l = np.fromfile(directory+f'skull_4p3l_410_565_530_yenotsu.DAT',dtype=np.float32).reshape(410,565,530)
skull_1p1l = np.fromfile(directory+f'skull_1p1l_410_565_530_yen.DAT',dtype=np.float32).reshape(410,565,530)

from scipy.ndimage import binary_dilation

def define_scalp_region(skull_mask, thickness=3.5, voxelsize=0.375):
    """
    Define the scalp area by dilating the skull mask.

    Args:
        skull_mask (3D numpy array): Binary mask where skull is 1 and background is 0.
        thickness (mm): Number of pixels to extend outward.

    Returns:
        scalp_mask (3D numpy array): Binary mask where the scalp area is 1.
    """
    # Create a structuring element for dilation (3D cube)
    structuring_element = np.ones((3, 3, 3))  # 3x3x3 cube, can be modified

    # Dilate the skull by the given thickness
    dilated_skull = binary_dilation(skull_mask, structure=structuring_element, iterations=int(thickness/voxelsize))

    # Subtract the original skull to get only the added layer (scalp)
    scalp_mask = dilated_skull.astype(np.uint8) - skull_mask.astype(np.uint8)

    return scalp_mask

border = define_scalp_region(skull_1p1l==0, 0.375)
skull_1p1l[border==1] = 0
skull_3p3l[border==1] = 0
    
def sampler(b, kappa, h):
    '''
    generate texture signal by gauss spectral function
    more details in paper
    Franceschini E, Mensah S, Amy D, Lefebvre JP. A 2-D anatomic breast ductal computer phantom for ultrasonic imaging. IEEE Trans Ultrason Ferroelectr Freq Control. 2006;53(7):1281-1288. doi:10.1109/tuffc.2006.1665076
    b: input whitenoise
    kappa: correlation length [mm]
    h: pixel size[mm]
    '''
    bhat = fft.fftn(b)
    nx, ny, nz = b.shape
    Lx = nx * h
    Ly = ny * h
    Lz = nz * h
    # Correct handling of even and odd dimensions
    def create_freq_vector(n, L):
        if n % 2 == 0:
            return np.concatenate((np.linspace(0, n // 2 - 1, n // 2), np.linspace(-n // 2, -1, n // 2))) * (2 * math.pi / L)
        else:
            return np.concatenate((np.linspace(0, (n - 1) // 2, (n + 1) // 2), np.linspace(-(n - 1) // 2, -1, (n - 1) // 2))) * (2 * math.pi / L)
    kz = create_freq_vector(nz, Lz)
    ky = create_freq_vector(ny, Ly)
    kx = create_freq_vector(nx, Lx)

    kkx, kky, kkz = np.meshgrid(ky, kx, kz)
    d2 = np.exp(-kappa * kappa * (np.power(kkx, 2) + np.power(kky, 2) + np.power(kkz, 2)) / 8)
    m = np.real(fft.ifftn(bhat * d2))
    return m

def sampling(mean, std=0.3, seed: int = None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        # Calculate the actual standard deviation
        actual_std = std * mean

        # Generate the sample using the chosen random number generator
        return rng.normal(mean, actual_std)
    else:
        return mean

    

def gen_acoustic(ph_type = 'rho', opt = 2, ph_ind = 1, water_bg = True, poro_select = None, seed = None):
    ac_nhp = np.zeros_like(head_phantom[:,:,:300])
    
    # if (opt == 2):
    #     poro_select = sto_poro[:,:,:300]
    # elif (opt ==3):
    #     poro_select = hu_poro[:,:,:300]
    
    if (ph_type == 'rho'):
        np.random.seed(ph_ind)
        if (opt == 0):
            tar_val = np.squeeze(fn_1p1l[:,0])
        else:
            tar_val = np.squeeze(fn_3p3l[:,0])
        if (opt > 1):
            para_water = 1000
            para_bone = sampling(2520, seed=seed)
            para_poro = poro_select*para_water+(1-poro_select)*para_bone # density
    elif (ph_type == 'cl'):
        np.random.seed(ph_ind+1000)
        if (opt == 0):
            tar_val = cl_1p1l
        else:
            tar_val = cl_3p3l
        if (opt > 1):
            para_water = 1.48
            para_bone = sampling(3.989, seed=seed)
            para_poro = para_water+(1-poro_select)*(para_bone-para_water)
    elif (ph_type == 'cs'):
        np.random.seed(ph_ind+1000)
        if (opt == 0):
            tar_val = cs_1p1l
        else:
            tar_val = cs_3p3l
        if (opt > 1):
            para_water = 1.48
            para_bone = sampling(3.989, seed=seed)
            para_poro = para_water+(1-poro_select)*(para_bone-para_water)
            np.random.seed(ph_ind+1000)
            para_poro /= sampling(2, seed=seed)
    elif (ph_type == 'alpha'):
        np.random.seed(ph_ind+3000)
        if (opt == 0):
            tar_val = fn_1p1l[:,3]
        else:
            tar_val = fn_3p3l[:,3]
        if (opt > 1):
            para_water = 0.0
            para_bone = sampling(0.8926, seed=seed)
            para_poro = para_water+(para_bone-para_water)*(poro_select**0.5)

    if (opt == 0):
        ac_nhp[skull_1p1l[:,:,:300]==1] = sampling(tar_val[1], seed=seed)
        ac_nhp[skull_1p1l[:,:,:300]==0] = tar_val[0]
        if not water_bg:
            ac_nhp[head_phantom[:,:,:300]==0] = 0.0
    elif (opt == 1):
        for i in range(1,6):
            ac_nhp[skull_3p3l[:,:,:300]==i] = tar_val[i]
            ac_nhp[skull_3p3l[:,:,:300]==0] = tar_val[0]
            if not water_bg:
                ac_nhp[head_phantom[:,:,:300]==0] = 0.0
    elif (opt == 2):
        for i in range(1,5):
            ac_nhp[skull_3p3l[:,:,:300]==i] = tar_val[i] #sampling(tar_val[i], seed=seed)
        ac_nhp[skull_3p3l[:,:,:300]==5] = para_poro[skull_3p3l[:,:,:300]==5]
        ac_nhp[skull_3p3l[:,:,:300]==0] = tar_val[0]
        if not water_bg:
            ac_nhp[head_phantom[:,:,:300]==0] = 0.0
    elif (opt == 3 or opt == 4):
        ac_nhp[skull_3p3l[:,:,:300]!=0] = para_poro[skull_3p3l[:,:,:300]!=0]
        ac_nhp[skull_3p3l[:,:,:300]==0] = tar_val[0]
        if not water_bg:
            ac_nhp[head_phantom[:,:,:300]==0] = 0.0

    return ac_nhp

#
fn_1p1l = np.fromfile(directory+f'param/fn_1p1l.DAT', dtype=np.float32).reshape(-1,4)
cl_1p1l = np.sqrt((fn_1p1l[:,1]+4*fn_1p1l[:,2]/3)/fn_1p1l[:,0])
cs_1p1l = np.sqrt(fn_1p1l[:,2]/fn_1p1l[:,0])
fn_3p3l = np.fromfile(directory+f'param/fn_4p3l.DAT', dtype=np.float32).reshape(-1,4)
fn_3p3l[1:5,:] = fn_3p3l[1,:]
cl_3p3l = np.sqrt((fn_3p3l[:,1]+4*fn_3p3l[:,2]/3)/fn_3p3l[:,0])
cs_3p3l = np.sqrt(fn_3p3l[:,2]/fn_3p3l[:,0])
nhp_opt = args.opt
ph_types = ['rho', 'cl', 'cs', 'alpha'] # rho, cl, cs, alph
Nx = 702
Ny = 702
Nz = 350

for ph_ind in [2]:
    with h5py.File(head_dir + f'head_{ph_ind}.h5', 'r') as f:
        head_phantom = f['data'][:]   # ← 关键：dataset 名字是 "data"
    np.random.seed(ph_ind)
    poro_select = None
    if (nhp_opt==2):
        ct_data = np.fromfile(directory+f'CT_375um_rotate_dim410_565_530_rot354_340.DAT',dtype=np.float32).reshape(410,565,530)
        max_hu = np.max(ct_data[:,:,:300])
        ct_data[skull_3p3l!=5] = 0 # v2 to leave only the diploe
        ct_data = sampler(ct_data, 0.375*4, 0.375) # v2 4 smoothing
        ct_data[skull_1p1l==0] = 0
        poro_select = 1-ct_data[:,:,:300]/max_hu
    elif (nhp_opt>2):
        ct_data = np.fromfile(directory+f'CT_375um_rotate_dim410_565_530_rot354_340.DAT',dtype=np.float32).reshape(410,565,530)
        if (nhp_opt==4):
            ct_data = sampler(ct_data, 0.375*4, 0.375)
        ct_data[skull_1p1l==0] = 0
        poro_select = 1-ct_data[:,:,:300]/np.max(ct_data[:,:,:300])
    mdic = {}
    for ph_type in ph_types:
        ac_nhp = gen_acoustic(ph_type, nhp_opt, ph_ind, poro_select=poro_select, seed=None) 
        mdic[ph_type] = ac_nhp
    
    file_name =  '/shared/anastasio-s3/CommonData/bozhen/'+f"acoustic/acoustic_distribution/nhp_{ph_ind}_op{nhp_opt}"
    if args.ver is not None:
        file_name += f"_v{args.ver}"
    # sio.savemat(file_name+".mat",mdic)

    acnhp = {}
    z_select = 180
    for ph_type in ph_types:
        nhp_full = np.ones((Nx, Ny, Nz), dtype=np.float32) * mdic[ph_type][0,0,0]
        nhp_full[round(Nx/2)-205:round(Nx/2)+205, round(Ny/2)-283:round(Ny/2)+282,:z_select] = mdic[ph_type][:,:,z_select-1::-1]
        acnhp[ph_type] = nhp_full
        
    print(acnhp.keys())
    sio.savemat(file_name+"_vifov.mat", acnhp)