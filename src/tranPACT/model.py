import numpy as np
from devito import (Grid, SubDomain,  SubDimension, 
                    Function, Eq, Inc, Operator, Abs)
from devito.builtins import initialize_function
from .utils import staggered_prop
import time
import warnings
import math


__all__ = ['GenericModel','TranPACTModel']

def downsample_index(n: int, stride: float):
    stride = float(stride)
    if stride <= 1.0:
        return np.arange(n, dtype=np.int64)
    n_new = int(math.ceil(n / stride))
    idx = np.round(np.linspace(0, n - 1, n_new)).astype(np.int64)
    return np.unique(idx)

def apply_downsample_3d(arr, ix, iy, iz):
    return arr[np.ix_(ix, iy, iz)]



class PhysicalDomain(SubDomain):
    """
    Used in defining GenericModel grid
    """
    name = 'physdomain'

    def __init__(self, so, fs=False):
        super(PhysicalDomain, self).__init__()
        self.so = so
        self.fs = fs

    def define(self, dimensions):
        map_d = {d: d for d in dimensions}
        if self.fs:
            map_d[dimensions[-1]] = ('middle', self.so, 0)
        return map_d

class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, opt_roi=None,
                 nbl=21, dtype=np.float32, subdomains=(), bcs="abc", grid=None):
        self.shape = shape
        self.space_order = space_order
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])
        # Default setup
        origin_pml = [dtype(o - s*nbl) for o, s in zip(origin, spacing)]
        shape_pml = np.array(shape) + 2 * self.nbl
        
        # Model size depending on freesurface
        physdomain = PhysicalDomain(space_order)
        subdomains = subdomains + (physdomain,)
        
        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        if grid is None:
            # Physical extent is calculated per cell, so shape - 1
            extent = tuple(np.array(spacing) * (shape_pml - 1))
            self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml,
                             dtype=dtype, subdomains=subdomains)
        else:
            self.grid = grid
            
        if opt_roi is None:
            self.opt_roi = np.ones(self.shape)
        else:
            self.opt_roi = opt_roi
    
    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))

class TranPACTModel(GenericModel):
    """
    The physical model used in seismic inversion processes.
    """
    _known_parameters = ['rho_x', 'rho_y', 'rho_z', 'alp_x', 'alp_y', 'alp_z',
                         'mu_diag', 'mu_xy', 'mu_yz', 'mu_xz', 'lam_diag']

    def __init__(self, origin, spacing, shape, space_order, 
                 opt_roi=None, nbl=21, dtype=np.float32,
                 subdomains=(), bcs="abc", grid=None,
                 medium_mode='mpml', medium_spatial=None, medium_geometry=None,
                 ct_data=None, cor_length=None,
                 medium_param=None,  water_index=None, texture_seed=None, use_static=True, use_downsample=False,
                 **kwargs):
        super(TranPACTModel, self).__init__(origin=origin, spacing=spacing, shape=shape,
                                            space_order=space_order, nbl=nbl,
                                            dtype=dtype, subdomains=subdomains,
                                            grid=grid, bcs=bcs, opt_roi=opt_roi)
        # Backward compatibility for 'medium_geo'
        if medium_geometry is None and 'medium_geo' in kwargs:
            medium_geometry = kwargs.pop('medium_geo')
            warnings.warn(
                "The 'medium_geo' argument is deprecated. Please use 'medium_geometry' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Backward compatibility for 'waterindex'
        if water_index is None and 'waterindex' in kwargs:
            water_index = kwargs.pop('waterindex')
            warnings.warn(
                "The 'waterindex' argument is deprecated. Please use 'water_index' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            
        self.medium_mode = medium_mode
        if self.medium_mode == 'true':
            if medium_spatial is None:
                raise ValueError("medium_spatial must be provided for 'true' medium mode.")
            try:
                cb = medium_spatial['cl'][0,0,0]
            except Exception as e:
                print("Error in medium_spatial: ", e)
                cb = np.sqrt(medium_param[water_index,1]/medium_param[water_index,0])
        elif self.medium_mode == 'aubry':
            if ct_data is None:
                raise ValueError("ct_data must be provided for 'aubry' medium mode.")
            if medium_param is None:
                raise ValueError("medium_param must be provided for 'aubry' medium mode.")
            if water_index is not None and water_index!=0:
                warnings.warn("water_index should be 0 for aubry method, using default value 0.")
                water_index = 0
            if cor_length is None:
                warnings.warn("cor_length is not provided, using default value 1.")
                cor_length = 1
            cb = np.sqrt(medium_param[water_index,1]/medium_param[water_index,0])
        elif self.medium_mode == 'mpml':
            if medium_geometry is None:
                raise ValueError("medium_geometry must be provided for 'mpml' medium mode.")
            if medium_param is None:
                raise ValueError("medium_param must be provided for 'mpml' medium mode.")
            cb = np.sqrt(medium_param[water_index,1]/medium_param[water_index,0])
        else:
            raise ValueError(f"Unknown medium mode: {self.medium_mode}. "
                             "Valid options are 'true', 'aubry', or 'mpml'.")

        self.geometry = medium_geometry
        self.ct_data = ct_data
        self.cor_length = cor_length
        self.use_downsample = use_downsample
        self.shape = shape
        # Initialize physics
        self._initialize_physics(space_order, **kwargs)
        self.initialize_medium(medium_param, water_index, texture_seed, medium_spatial=medium_spatial, 
            use_static=use_static, use_downsample=self.use_downsample,**kwargs)
        self._initialize_damp(nbl, cb)
    
    def _initialize_physics(self, so, **kwargs):
        """
        Initialize physical parameters and type of physics from inputs.
        """
        starttime = time.time()
        x, y, z = self.grid.dimensions
        # Initialize the input physical parameters
        self.alp_x = Function(name='alp_x0', grid=self.grid, space_order=so, parameter=True, staggered=(-x,), dtype=np.float32)
        self.alp_y = Function(name='alp_y0', grid=self.grid, space_order=so, parameter=True, staggered=(y,), dtype=np.float32)
        self.alp_z = Function(name='alp_z0', grid=self.grid, space_order=so, parameter=True, staggered=(z,), dtype=np.float32)
        self.b_x = Function(name='bx0', grid=self.grid, space_order=so, parameter=True, staggered=(-x,), dtype=np.float32)
        self.b_y = Function(name='by0', grid=self.grid, space_order=so, parameter=True, staggered=(y,), dtype=np.float32)
        self.b_z = Function(name='bz0', grid=self.grid, space_order=so, parameter=True, staggered=(z,), dtype=np.float32)
        self.mu_diag = Function(name='mu_diag0', grid=self.grid, space_order=so, parameter=True, staggered=(), dtype=np.float32)
        self.mu_xy = Function(name='mu_xy0', grid=self.grid, space_order=so, parameter=True, staggered=(-x,y), dtype=np.float32)
        self.mu_yz = Function(name='mu_yz0', grid=self.grid, space_order=so, parameter=True, staggered=(y,z), dtype=np.float32)
        self.mu_xz = Function(name='mu_xz0', grid=self.grid, space_order=so, parameter=True, staggered=(-x,z), dtype=np.float32)
        self.lam_diag = Function(name='lam_diag0', grid=self.grid, space_order=so, parameter=True, staggered=(), dtype=np.float32)
        middle = time.time()
        if 'print' in kwargs:
            print(f'It takes {middle-starttime} second to set medium functions')

    def initialize_medium(
        self,
        medium_param,
        water_index=2,
        texture_seed=None,
        medium_spatial=None,
        use_static=True, use_downsample=False,
        **kwargs
    ):
        """
        Instrumented version of initialize_medium:
        - DOES NOT change any numerical logic
        - ONLY adds timing / logging
        """

        import time
        cor_length = kwargs.get('cor_length', self.cor_length)
        verbose = kwargs.get('print', True)

        t0 = time.time()
        if verbose:
            dist = getattr(self.grid, "distributor", None)
            print(
                f"[INIT_MEDIUM] ENTER initialize_medium | "
                f"cor_length={cor_length}",
                flush=True
            )
            print(
#                 f"[INIT_MEDIUM DBG] grid.shape={self.grid.shape} model.shape={self.shape} "
                f"nprocs={getattr(dist, 'nprocs', 'NA')} "
                f"myrank={getattr(dist, 'myrank', 'NA')}",
                flush=True
            )
            print(
#                 f"[INIT_MEDIUM DBG] alp_x.data.shape={self.alp_x.data.shape} "
                f"b_x.data.shape={self.b_x.data.shape}",
                flush=True
            )
            # print(
            #     f"[INIT_MEDIUM DBG] distributor attrs: "
            #     f"glb_pos_map={getattr(dist, 'glb_pos_map', 'NA')} "
            #     f"glb_numb={getattr(dist, 'glb_numb', 'NA')} "
            #     f"shape={getattr(dist, 'shape', 'NA')} "
            #     f"dimensions={getattr(dist, 'dimensions', 'NA')}",
            #     flush=True
            # )
        # ------------------------------------------------------------
        # 1. Aubry / staggered convolution (最慢的部分)
        # ------------------------------------------------------------
        t1 = time.time()
        if verbose:
            print("[INIT_MEDIUM] ENTER staggered_prop", flush=True)

        if use_static:
            if verbose:
                print("[INIT_MEDIUM] USE Static prop", flush=True)

            (
                med_rho_x, med_rho_y, med_rho_z,
                med_alpha_x, med_alpha_y, med_alpha_z,
                med_mu_diag, med_mu_xy, med_mu_yz, med_mu_xz,
                med_lam_diag
            ) = staggered_prop(
                self.geometry,
                medium_param,
                water_index,
                texture_seed,
                medium_spatial=medium_spatial,
                ct_data=self.ct_data,
                cor_length=cor_length, use_ct_smooth=True,
                ct_smooth_path = (
                "/shared/anastasio-s3/CommonData/bozhen/gfjr_data/static/"
                "ct_smooth_cor1.npz"
                )
            )

        else:
            if verbose:
                print("[INIT_MEDIUM] USE sampler prop", flush=True)

            (
                med_rho_x, med_rho_y, med_rho_z,
                med_alpha_x, med_alpha_y, med_alpha_z,
                med_mu_diag, med_mu_xy, med_mu_yz, med_mu_xz,
                med_lam_diag
            ) = staggered_prop(
                self.geometry,
                medium_param,
                water_index,
                texture_seed,
                medium_spatial=medium_spatial,
                ct_data=self.ct_data,
                cor_length=cor_length, use_ct_smooth=False
            )
        
        if use_downsample:
            # model physical shape (small grid)
            Nx, Ny, Nz = self.shape

            # full property shape
            fx, fy, fz = med_rho_x.shape

            # 推断 stride（full → small）
            sx = fx / Nx
            sy = fy / Ny
            sz = fz / Nz

            ix = downsample_index(fx, sx)
            iy = downsample_index(fy, sy)
            iz = downsample_index(fz, sz)

            med_rho_x   = apply_downsample_3d(med_rho_x,   ix, iy, iz)
            med_rho_y   = apply_downsample_3d(med_rho_y,   ix, iy, iz)
            med_rho_z   = apply_downsample_3d(med_rho_z,   ix, iy, iz)

            med_alpha_x = apply_downsample_3d(med_alpha_x, ix, iy, iz)
            med_alpha_y = apply_downsample_3d(med_alpha_y, ix, iy, iz)
            med_alpha_z = apply_downsample_3d(med_alpha_z, ix, iy, iz)

            med_mu_diag = apply_downsample_3d(med_mu_diag, ix, iy, iz)
            med_mu_xy   = apply_downsample_3d(med_mu_xy,   ix, iy, iz)
            med_mu_yz   = apply_downsample_3d(med_mu_yz,   ix, iy, iz)
            med_mu_xz   = apply_downsample_3d(med_mu_xz,   ix, iy, iz)

            med_lam_diag = apply_downsample_3d(med_lam_diag, ix, iy, iz)
            print("Downsampled med_rho_x shape", med_rho_x.shape)
        else:
            print("Use original med_rho_x shape", med_rho_x.shape)
        t2 = time.time()
        if verbose:
            print(
                f"[INIT_MEDIUM] EXIT staggered_prop | "
                f"dt={t2 - t1:.3f} s",
                flush=True
            )

        # ------------------------------------------------------------
        # 2. Write fields into Devito Functions
        # ------------------------------------------------------------
        nbl = self.nbl
        if verbose:
            print("[INIT_MEDIUM] ENTER initialize_function (Devito fields)", flush=True)
        
        print("med_rho_x shape =", med_rho_x.shape)

        t3 = time.time()
#         print(f"[INIT_MEDIUM DBG] BEFORE init alp_x: func={self.alp_x.data.shape} input={med_alpha_x.shape}", flush=True)
        initialize_function(self.alp_x, med_alpha_x, nbl)
#         print(f"[INIT_MEDIUM DBG] AFTER  init alp_x: func={self.alp_x.data.shape}", flush=True)
        initialize_function(self.alp_y, med_alpha_y, nbl)
        initialize_function(self.alp_z, med_alpha_z, nbl)
        initialize_function(self.b_x, 1.0 / med_rho_x, nbl)
        initialize_function(self.b_y, 1.0 / med_rho_y, nbl)
        initialize_function(self.b_z, 1.0 / med_rho_z, nbl)
        initialize_function(self.mu_diag, med_mu_diag, nbl)
        initialize_function(self.mu_xy, med_mu_xy, nbl)
        initialize_function(self.mu_yz, med_mu_yz, nbl)
        initialize_function(self.mu_xz, med_mu_xz, nbl)
        initialize_function(self.lam_diag, med_lam_diag, nbl)

        t4 = time.time()
        if verbose:
            print(
                f"[INIT_MEDIUM] EXIT initialize_function | "
                f"dt={t4 - t3:.3f} s",
                flush=True
            )

        # ------------------------------------------------------------
        # 3. Total
        # ------------------------------------------------------------
        if verbose:
            print(
                f"[INIT_MEDIUM] EXIT initialize_medium | "
                f"total dt={t4 - t0:.3f} s",
                flush=True
            )

    # def initialize_medium(self, medium_param, water_index=2, texture_seed=None, medium_spatial=None, **kwargs):
    #     # setting medium parameters using self implemented averaging
    #     cor_length=kwargs.get('cor_length', self.cor_length)
    #     starttime = time.time()
    #     med_rho_x, med_rho_y, med_rho_z, med_alpha_x, med_alpha_y, med_alpha_z, med_mu_diag, med_mu_xy, med_mu_yz, med_mu_xz, med_lam_diag = staggered_prop(self.geometry, medium_param, water_index, texture_seed, medium_spatial=medium_spatial, ct_data=self.ct_data, cor_length=cor_length)
    #     middle = time.time()
    #     if 'print' in kwargs:
    #         print(f'It takes {middle-starttime} second to do convolution')
    #     nbl = self.nbl
    #     initialize_function(self.alp_x, med_alpha_x, nbl)
    #     initialize_function(self.alp_y, med_alpha_y, nbl)
    #     initialize_function(self.alp_z, med_alpha_z, nbl)
    #     initialize_function(self.b_x, 1/med_rho_x, nbl)
    #     initialize_function(self.b_y, 1/med_rho_y, nbl)
    #     initialize_function(self.b_z, 1/med_rho_z, nbl)
    #     initialize_function(self.mu_diag, med_mu_diag, nbl)
    #     initialize_function(self.mu_xy, med_mu_xy, nbl)
    #     initialize_function(self.mu_yz, med_mu_yz, nbl)
    #     initialize_function(self.mu_xz, med_mu_xz, nbl)
    #     initialize_function(self.lam_diag, med_lam_diag, nbl)
    #     endtime = time.time()
    #     if 'print' in kwargs:
    #         print(f'It takes {endtime-middle} second to set the values')
        
    def _initialize_damp(self, nbl, cb):
        
        x, y, z = self.grid.dimensions
        dx,_,_ = self.spacing
        
        # Split Coordinate pml 
        n = 2
        R_ref= 10**-6
        sigma_max = -np.log(R_ref) * (n + 1) * cb / (2 * nbl * dx)

        self.s_x = Function(name='bc_x', dimensions=(x,), shape=(self.grid.shape[0],),
                       grid=self.grid, parameter=True, dtype=self.dtype)
        eqs = [Eq(self.s_x, 0)]
        dim_lx = SubDimension.left(name='pml_lx', parent=x, thickness=nbl)
        pos_lx = Abs(nbl - (dim_lx - x.symbolic_min)) / float(nbl)
        val = sigma_max * np.power(pos_lx, n)
        eqs += [Inc(self.s_x.subs({x: dim_lx}), val)]
        dim_hx = SubDimension.right(name='pml_hx', parent=x, thickness=nbl)
        pos_hx = Abs(nbl - (x.symbolic_max - dim_hx)) / float(nbl)
        val = sigma_max * np.power(pos_hx, n)
        eqs += [Inc(self.s_x.subs({x: dim_hx}), val)]

        self.s_y = Function(name='bc_y', dimensions=(y,), shape=(self.grid.shape[1],),
                       grid=self.grid, parameter=True, dtype=self.dtype)
        eqs += [Eq(self.s_y, 0)]
        dim_ly = SubDimension.left(name='pml_ly', parent=y, thickness=nbl)
        pos_ly = Abs(nbl - (dim_ly - y.symbolic_min)) / float(nbl)
        val = sigma_max * np.power(pos_ly, n)
        eqs += [Inc(self.s_y.subs({y: dim_ly}), val)]
        dim_hy = SubDimension.right(name='pml_hy', parent=y, thickness=nbl)
        pos_hy = Abs(nbl - (y.symbolic_max - dim_hy)) / float(nbl)
        val = sigma_max * np.power(pos_hy, n)
        eqs += [Inc(self.s_y.subs({y: dim_hy}), val)]

        self.s_z = Function(name='bc_z', dimensions=(z,), shape=(self.grid.shape[2],),
                       grid=self.grid, parameter=True, dtype=self.dtype)
        eqs += [Eq(self.s_z, 0)]
        dim_lz = SubDimension.left(name='pml_lz', parent=z, thickness=nbl)
        pos_lz = Abs(nbl - (dim_lz - z.symbolic_min)) / float(nbl)
        val = sigma_max * np.power(pos_lz, n)
        eqs += [Inc(self.s_z.subs({z: dim_lz}), val)]
        dim_hz = SubDimension.right(name='pml_hz', parent=z, thickness=nbl)
        pos_hz = Abs(nbl - (z.symbolic_max - dim_hz)) / float(nbl)
        val = sigma_max * np.power(pos_hz, n)
        eqs += [Inc(self.s_z.subs({z: dim_hz}), val)]
        Operator(eqs, name='initpml')()
