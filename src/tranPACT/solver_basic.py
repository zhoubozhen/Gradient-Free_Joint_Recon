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

        dist = getattr(model.grid, "distributor", None)
        rec_pos_arr = np.asarray(rec_pos, dtype=model.dtype)

        local_rec_pos = np.ascontiguousarray(rec_pos_arr, dtype=model.dtype)
        if dist is not None and getattr(dist, "nprocs", 1) > 1:
            myrank = getattr(dist, "myrank", 0)
            # print(f"[SPARSE BUILD] rank={myrank} use full rec_pos once, shape={local_rec_pos.shape}", flush=True)
            # print(f"[SPARSE BUILD] rank={myrank} local_rec_pos.first={local_rec_pos[:3]}", flush=True)
            # print(f"[SPARSE BUILD] rank={myrank} local_rec_pos.min={local_rec_pos.min(axis=0)} max={local_rec_pos.max(axis=0)}", flush=True)

        if rec is None:
            self.rec = Receiver(
                name="rec",
                grid=model.grid,
                npoint=local_rec_pos.shape[0],
                time_range=self.time_range,
                coordinates=local_rec_pos,
                dtype=model.dtype
            )
        else:
            self.rec = rec
        if src is None:
            self.src = PointSource(
                name='src',
                grid=model.grid,
                npoint=local_rec_pos.shape[0],
                time_range=self.time_range,
                coordinates=local_rec_pos,
                dtype=model.dtype
            )
        else:
            self.src = src

        dist = getattr(model.grid, "distributor", None)
        # print(
        #     f"[SPARSE DBG] rank={getattr(dist, 'myrank', 'NA')} "
        #     f"rec.data.shape={self.rec.data.shape} "
        #     f"src.data.shape={self.src.data.shape} "
        #     f"rec.coords.shape={self.rec.coordinates.data.shape} "
        #     f"src.coords.shape={self.src.coordinates.data.shape}",
        #     flush=True
        # )
        try:
            # print(
            #     f"[SPARSE DBG] rank={getattr(dist, 'myrank', 'NA')} "
            #     f"rec.coords.first={self.rec.coordinates.data[:3]} "
            #     f"src.coords.first={self.src.coordinates.data[:3]}",
            #     flush=True
            # )
            # print(
            #     f"[SPARSE DBG] rank={getattr(dist, 'myrank', 'NA')} "
            #     f"rec.coords._data.shape={self.rec.coordinates._data.shape} "
            #     f"src.coords._data.shape={self.src.coordinates._data.shape}",
            #     flush=True
            # )
            # print(
            #     f"[SPARSE DBG] rank={getattr(dist, 'myrank', 'NA')} "
            #     f"rec.coords._data.first={self.rec.coordinates._data[:3]} "
            #     f"src.coords._data.first={self.src.coordinates._data[:3]}",
            #     flush=True
            # )
            # print(
            #     f"[SPARSE DBG] rank={getattr(dist, 'myrank', 'NA')} "
            #     f"rec.coords._data.min={self.rec.coordinates._data.min(axis=0)} "
            #     f"rec.coords._data.max={self.rec.coordinates._data.max(axis=0)}",
            #     flush=True
            # )
            rank_tag = getattr(dist, 'myrank', 0)
            np.save(f"/tmp/rec_coords_rank{rank_tag}.npy", self.rec.coordinates._data.copy())
            np.save(f"/tmp/src_coords_rank{rank_tag}.npy", self.src.coordinates._data.copy())
            # print(f"[SPARSE DBG] dumped /tmp/rec_coords_rank{rank_tag}.npy", flush=True)
        except Exception as e:
            pass
            # print(f"[SPARSE DBG] coords print failed: {e}", flush=True)

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
        import time
        t_obj0 = time.time()
        # print(f"[ADJ TIME] BEFORE create v/tau rank={getattr(model.grid.distributor, 'myrank', 'NA')}", flush=True)
        v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order,
                           time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, 
                             time_order=self.time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        # print(f"[ADJ TIME] AFTER  create v/tau rank={getattr(model.grid.distributor, 'myrank', 'NA')} dt={time.time()-t_obj0:.3f}s", flush=True)
        
        # Setting initial pressure
        if math.prod(p0_flat.shape)==math.prod(model.shape):
            p0 = p0_flat.copy().reshape(model.shape)
        else:
            p0 = np.zeros(model.shape,dtype=np.float32)
            p0[opt_roi>0] = p0_flat.ravel()

        # print(f"[FORWARD DBG] p0.shape={p0.shape} tau00.data.shape={tau[0,0].data.shape} target_slice=({Nx},{Ny},{Nz})", flush=True)
        tau[0,0].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0
        tau[1,1].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0
        tau[2,2].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0
        
        # Execute operator and return wavefield and receiver data 
        # can use print(op_fwd.parameters) to check valid apply input
        if __import__("os").environ.get("DUMMY_MPI_FORWARD", "0") == "1":
            # print("[FORWARD DBG] DUMMY_MPI_FORWARD=1 -> skip op_fwd.apply and return zeros", flush=True)
            return np.zeros(rec.data.shape, dtype=np.float32).ravel()
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

    def adjoint(self, rec_data_flat, src=None, v=None, model=None, fullscale=False, rec_start=None, rec_end=None, **kwargs):
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
        rec_data = np.asarray(rec_data_flat).reshape(self.rec.data.shape[0], -1)
        measured_pressure = np.asarray(rec_data, dtype=np.float32, order='C')
        measured_pressure = np.ascontiguousarray(measured_pressure)

        # print(f"[ADJOINT DBG] rec_data_flat shape={np.shape(rec_data_flat)} dtype={getattr(rec_data_flat, 'dtype', type(rec_data_flat))}", flush=True)
        # print(f"[ADJOINT DBG] rec_data shape={rec_data.shape} dtype={rec_data.dtype} C={rec_data.flags['C_CONTIGUOUS']}", flush=True)
        # print(f"[ADJOINT DBG] measured_pressure shape={measured_pressure.shape} dtype={measured_pressure.dtype} C={measured_pressure.flags['C_CONTIGUOUS']}", flush=True)
        # print(f"[ADJOINT DBG] src.data shape={src.data.shape} dtype={src.data.dtype} C={src.data.flags['C_CONTIGUOUS']}", flush=True)

        assert measured_pressure.shape == src.data.shape, f"shape mismatch: measured_pressure {measured_pressure.shape} vs src.data {src.data.shape}"

        # print(f"[ADJOINT DBG] write raw local buffer src._data with shape={src._data.shape}", flush=True)
        src._data[:, :] = measured_pressure
        # print(rec_data_flat.shape)
        # print(self.rec.data.shape)
        
        # Execute operator and return wavefield
        # can use print(op_fwd.parameters) to check valid apply input
        if __import__("os").environ.get("DUMMY_MPI_ADJOINT", "0") == "1":
            # print("[ADJOINT DBG] DUMMY_MPI_ADJOINT=1 -> skip op_adj.apply and return zeros", flush=True)
            return np.zeros((Nx, Ny, Nz), dtype=np.float32)
        import time
        t_build0 = time.time()
        # print(f"[ADJ TIME] BEFORE build op_adj rank={getattr(model.grid.distributor, 'myrank', 'NA')}", flush=True)
        op_adj = self.op_adj()
        # print(f"[ADJ TIME] AFTER  build op_adj rank={getattr(model.grid.distributor, 'myrank', 'NA')} dt={time.time()-t_build0:.3f}s", flush=True)
        import cupy as cp
        cp.cuda.Device().synchronize()
        t_apply0 = time.time()
        # print(f"[ADJ TIME] BEFORE op_adj.apply rank={getattr(model.grid.distributor, 'myrank', 'NA')}", flush=True)
        op_adj.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],
                     sigma_xy=tau[0,1], sigma_xz=tau[0,2], sigma_yy=tau[1,1],
                     sigma_yz=tau[1,2], sigma_zz=tau[2,2],
                     time_m=0, time_M=self.time_range.num-1, **kwargs)
        cp.cuda.Device().synchronize()
        # print(f"[ADJ TIME] AFTER  op_adj.apply rank={getattr(model.grid.distributor, 'myrank', 'NA')} dt={time.time()-t_apply0:.3f}s wall={time.time():.6f}", flush=True)
        # print(f"[ADJOINT DBG] tau00.data.shape={tau[0,0].data.shape} fullscale={fullscale} target_slice=({Nx},{Ny},{Nz})", flush=True)
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
        return self.forward(p0_flat).ravel()
        
    def mpi_adjoint(self, p0_ravel, comm, Nx, Ny, Nz, fullscale=False):
        import numpy as np

        rank = comm.Get_rank()
        local_nrec = self.src.data.shape[1]
        arr = np.asarray(p0_ravel, dtype=np.float32)

        if arr.ndim == 1:
            nt = self.src.data.shape[0]
            global_nrec = arr.size // nt
            arr2 = arr.reshape(nt, global_nrec)
        else:
            arr2 = arr

        if arr2.shape[1] == local_nrec:
            local_arr = np.ascontiguousarray(arr2, dtype=np.float32)
        else:
            start = rank * local_nrec
            end = start + local_nrec
            local_arr = np.ascontiguousarray(arr2[:, start:end], dtype=np.float32)

        print(f"[MPI_ADJOINT DBG] rank={rank} input shape={arr2.shape} local_nrec={local_nrec} local_arr={local_arr.shape}", flush=True)
        return self.adjoint(local_arr.ravel(), fullscale=False, rec_start=start, rec_end=end)
