import numpy as np
from devito import (VectorTimeFunction, TensorTimeFunction)
from devito.tools import memoized_meth
from .operator import ForwardOperator, AdjointOperator
import math
from .seismic_util import TimeAxis, PointSource, Receiver

__all__ = ['TranPACTWaveSolver']

class TranPACTWaveSolver(object):
    """
    Initializes the wave solver for the TranPACT model.

    This class handles the forward and adjoint wave simulations for a given
    TranPACT model.  It manages source and receiver configurations,
    time stepping, and other simulation parameters.  The solver
    uses cached operators for efficiency.
    """

    def __init__(self, model, rec=None, src=None, time_range=None, time_order=2, rec_pos=None, Nt=None, dt=None, **kwargs):
        """
        The initialization of this class has evolved over time to provide more
        flexibility.  There are two main ways to initialize it:

        **Older Version:**
        ```python
        solver = TranPACTWaveSolver(model, rec, src, time_range, p0, time_order)
        ```
        Where:
            model:      The TranPACTModel instance.
            rec:        A Receiver object.  If None, a default Receiver is created using rec_pos.
            src:        A PointSource object. If None, a default PointSource is created using rec_pos.
            time_range: A TimeAxis object. If None, a default TimeAxis is created using Nt and dt.
            p0:         Initial pressure distribution.
            time_order: The time order of the simulation (default: 2).

        **Newer Version:**
         ```python
        solver = TranPACTWaveSolver(model, rec_pos=rec_pos, Nt=Nt, dt=dt, to=to)
        ```
        Where:
            model:  The TranPACTModel instance.
            rec_pos: The receiver coordinates (NumPy array).
            Nt:     The number of time steps.
            dt:     The time step size.
            to:     The time order (usually 2).

        The newer version excludes the need of defining receiver and source outside the solver.
        This help to avoid confusion and makes the code cleaner. The older version is still supported
        for backward compatibility, but the newer version is recommended for new code.
        """
        self.model = model
        self.space_order = self.model.space_order
        self.time_range = time_range or TimeAxis(start=0, num=Nt, step=dt)
        if rec is None:
            self.rec = Receiver(name="rec", grid=model.grid, npoint=rec_pos.shape[0],
                                time_range=self.time_range, dtype=model.dtype)
            self.rec.coordinates.data[:,:] = rec_pos
        else:
            self.rec = rec
        if src is None:
            self.src = PointSource(name='src', grid=model.grid, npoint=rec_pos.shape[0],
                                   time_range=self.time_range, dtype=model.dtype)
            self.src.coordinates.data[:,:] = rec_pos
        else:
            self.src = src
        self.p0_roi = (np.zeros(model.shape, dtype=np.float32))[self.model.opt_roi>0]
        self.time_order = time_order
        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self):
        """Cached operator for forward runs"""
        return ForwardOperator(self.model, p0=self.p0_roi, rec=self.rec,
                               space_order=self.space_order,
                               time_order =self.time_order, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, rec=self.rec, src=self.src,
                               space_order=self.space_order,
                               time_order =self.time_order, **self._kwargs)

    def forward(self, p0_flat, src=None, rec=None, model=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        model : Model, optional
            Object containing the physical parameters.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        DTYPE = model.dtype
        nbl = model.nbl
        opt_roi = model.opt_roi
        (Nx,Ny,Nz) = model.shape
        
        # Setting new time function as initialization
        v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order,
                           time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, 
                             time_order=self.time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        
        # Setting initial pressure
        if math.prod(p0_flat.shape)==math.prod(model.shape):
            p0 = p0_flat.copy().reshape(model.shape)
        else:
            p0 = np.zeros(model.shape,dtype=np.float32)
            p0[opt_roi>0] = p0_flat.ravel()
        tau[0,0].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0
        tau[1,1].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0
        tau[2,2].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0
        
        # Execute operator and return wavefield and receiver data 
        # can use print(op_fwd.parameters) to check valid apply input
        op_fwd = self.op_fwd() # acquiring cached operator
        op_fwd.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],
                     sigma_xy=tau[0,1], sigma_xz=tau[0,2], sigma_yy=tau[1,1],
                     sigma_yz=tau[1,2], sigma_zz=tau[2,2],
                     time_m=0, time_M=self.time_range.num-1, **kwargs)

        if type(rec)==type([]):
            rec_bundle_data = []
            for rind in range(len(rec)):
                rec_bundle_data.append(rec[rind].data.ravel())
            return rec_bundle_data
        else:
            return rec.data.ravel().copy()

    def adjoint(self, rec_data_flat, src=None, v=None, model=None, fullscale=False, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        DTYPE = model.dtype
        nbl = model.nbl
        opt_roi = model.opt_roi
        (Nx,Ny,Nz) = model.shape
        
        # Setting new time function as initialization
        v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order,
                           time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, 
                             time_order=self.time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        
        # Setting received data as adjoint source
        rec_data = rec_data_flat.reshape(self.rec.data.shape[0],-1)
        measured_pressure = np.zeros(rec_data.shape, dtype=np.float32)
        measured_pressure += rec_data
        src.data[:,:] = measured_pressure
        # print(rec_data_flat.shape)
        # print(self.rec.data.shape)
        
        # Execute operator and return wavefield
        # can use print(op_fwd.parameters) to check valid apply input
        op_adj = self.op_adj()
        op_adj.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],
                     sigma_xy=tau[0,1], sigma_xz=tau[0,2], sigma_yy=tau[1,1],
                     sigma_yz=tau[1,2], sigma_zz=tau[2,2],
                     time_m=0, time_M=self.time_range.num-1, **kwargs)
        if fullscale:
            p0 = -(tau[0,0].data[2,:,:,:]+tau[1,1].data[2,:,:,:]
                +tau[2,2].data[2,:,:,:])/3
        else:
            p0 = -(tau[0,0].data[2,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz]
                +tau[1,1].data[2,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz]
                +tau[2,2].data[2,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz])/3
        del measured_pressure
        # p0[opt_roi==0] = 0.0
        return p0
    
    def devito_mpi_collect(self, array, comm, Nx, Ny, Nz):
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size>1:
            if array.ndim>3:
                composed = np.zeros((array.shape[0],Nx,Ny,Nz),dtype=np.float32)
                for ind in range(array.shape[0]):
                    composed[ind,:,:,:] = self.devito_mpi_collect(array[ind,:,:,:], comm, Nx, Ny, Nz)
                return composed
            else:
                newData = comm.gather(array,root=0)
                if rank==0:
                    newData = np.array(newData,dtype=np.float32)
                    if size==4:
                        compose_0 = np.concatenate((newData[0,:,:,:],newData[1,:,:,:]),axis=1)
                        compose_1 = np.concatenate((newData[2,:,:,:],newData[3,:,:,:]),axis=1)
                        composed = np.concatenate((compose_0,compose_1),axis=0)
                    elif size==2:
                        composed = np.concatenate((newData[0,:,:,:],newData[1,:,:,:]),axis=0)
                else:
                    composed = np.zeros((Nx,Ny,Nz),dtype=np.float32)
                comm.Bcast(composed, root=0)
                return composed
        else:
            return array

    def mpi_forward(self, p0_flat, comm, Nt):
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size>1:
            p0_forward_T = self.forward(p0_flat).reshape(Nt, -1).T.ravel()
            newData = comm.gather(p0_forward_T,root=0)
            if rank==0:
                p0_forward_T = np.array(newData,dtype=np.float32).ravel()
            else:
                p0_forward_T = np.zeros(p0_forward_T.shape[0]*size,dtype=np.float32)
            del newData
            comm.Bcast(p0_forward_T, root=0)
            p0_forward = p0_forward_T.reshape(-1, Nt).T
            return p0_forward.ravel()
        else:
            return self.forward(p0_flat).ravel()
        
    def mpi_adjoint(self, p0_ravel, comm, Nx, Ny, Nz, fullscale=True):
        p0_adj = self.adjoint(p0_ravel, fullscale=True)
        nbl = self.model.nbl
        p0_combined = self.devito_mpi_collect(p0_adj, comm, Nx+2*nbl, Ny+2*nbl, Nz+2*nbl)
        if fullscale:
            return p0_combined
        else:
            return p0_combined[nbl:-nbl, nbl:-nbl, nbl:-nbl]