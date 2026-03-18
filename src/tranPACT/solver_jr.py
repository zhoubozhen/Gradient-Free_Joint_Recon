import numpy as np
from devito import (DevitoCheckpoint, CheckpointOperator, Revolver,
                    Function, VectorTimeFunction, TensorTimeFunction,
                    Eq, Operator, diag, grad, div)
from devito.tools import memoized_meth
from .operator import JRForwardOperator, JRAdjointOperator, JRAdjointcheckOperator
from .seismic_util import TimeAxis, Receiver, PointSource
from .solver_basic import TranPACTWaveSolver

__all__ = ['TranPACTJRWaveSolver']

class TranPACTJRWaveSolver(TranPACTWaveSolver):
    @memoized_meth
    def op_jrfwd(self):
        """Cached operator for forward runs"""
        return JRForwardOperator(self.model, rec=self.rec,
                                 space_order=self.space_order,
                                 time_order=self.time_order, **self._kwargs)
    
    @memoized_meth
    def op_jradj(self):
        """Cached operator for forward runs"""
        return JRAdjointOperator(self.model, rec=self.rec, src=self.src,
                                 space_order=self.space_order,
                                 time_order=self.time_order, **self._kwargs)
    
    @memoized_meth
    def op_jradjcp(self, checkpoint=False):
        """Cached operator for forward runs"""
        return JRAdjointcheckOperator(self.model, rec=self.rec, src=self.src,
                                      space_order=self.space_order, time_order=self.time_order,
                                      checkpoint=checkpoint, **self._kwargs)

    def state_assign(self, tau, v, tindex, state_init, **kwargs):
        '''
        Compatible w/wo the use of MPI
        '''
        tau[0,0].data[tindex,:,:,:] = state_init[0,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[1,1].data[tindex,:,:,:] = state_init[1,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[2,2].data[tindex,:,:,:] = state_init[2,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[0,1].data[tindex,:,:,:] = state_init[3,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[1,0].data[tindex,:,:,:] = state_init[3,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[1,2].data[tindex,:,:,:] = state_init[4,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[2,1].data[tindex,:,:,:] = state_init[4,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[0,2].data[tindex,:,:,:] = state_init[5,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        tau[2,0].data[tindex,:,:,:] = state_init[5,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        v[0].data[tindex,:,:,:] = state_init[6,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        v[1].data[tindex,:,:,:] = state_init[7,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
        v[2].data[tindex,:,:,:] = state_init[8,:,:,:].copy() if isinstance(state_init, np.ndarray) else state_init
    
    def state_extract(self, tau, v, tindex):
        state_init = np.zeros([9]+list(tau[0,0].data[0,:,:,:].shape),dtype=np.float32)
        state_init[0,:,:,:] = tau[0,0].data[tindex,:,:,:]
        state_init[1,:,:,:] = tau[1,1].data[tindex,:,:,:]
        state_init[2,:,:,:] = tau[2,2].data[tindex,:,:,:]
        state_init[3,:,:,:] = tau[0,1].data[tindex,:,:,:]
        state_init[4,:,:,:] = tau[1,2].data[tindex,:,:,:]
        state_init[5,:,:,:] = tau[0,2].data[tindex,:,:,:]
        state_init[6,:,:,:] = v[0].data[tindex,:,:,:]
        state_init[7,:,:,:] = v[1].data[tindex,:,:,:]
        state_init[8,:,:,:] = v[2].data[tindex,:,:,:]
        return state_init

    def save_check_legacy(self, state_fwd_init, Nt, rec=None, model=None, v=None, tau=None, n_gpucap=24):
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        op_fwd = self.op_fwd()
        DTYPE = model.dtype

        if (Nt-1)%n_gpucap ==0:
            n_gpucap = n_gpucap-1
            if (Nt-1)%n_gpucap ==0:
                num_check = (Nt-1)//n_gpucap
            else:
                num_check = (Nt-1)//n_gpucap+1
        else:
            num_check = (Nt-1)//n_gpucap+1
        print('Number of checkpoints: {:d}'.format(num_check))
        check_save = np.zeros(np.concatenate((np.array([num_check,9]),np.array(model.shape)+32)),dtype=np.float32)
        
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        
        self.state_assign(tau, v, 0, state_fwd_init)

        for cind in range(num_check):
            tindex = (cind*n_gpucap) % 3
            check_save[cind,0,:,:,:] = tau[0,0].data[tindex,:,:,:]
            check_save[cind,1,:,:,:] = tau[1,1].data[tindex,:,:,:]
            check_save[cind,2,:,:,:] = tau[2,2].data[tindex,:,:,:]
            check_save[cind,3,:,:,:] = tau[0,1].data[tindex,:,:,:]
            check_save[cind,4,:,:,:] = tau[1,2].data[tindex,:,:,:]
            check_save[cind,5,:,:,:] = tau[0,2].data[tindex,:,:,:]
            check_save[cind,6,:,:,:] = v[0].data[tindex,:,:,:]
            check_save[cind,7,:,:,:] = v[1].data[tindex,:,:,:]
            check_save[cind,8,:,:,:] = v[2].data[tindex,:,:,:]
            if cind!=(num_check-1):
                op_fwd.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                             sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                             sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], rec=rec,
                             time_m=cind*n_gpucap, time_M=(cind+1)*n_gpucap-1)
            else:
                op_fwd.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                             sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                             sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], rec=rec,
                             time_m=cind*n_gpucap, time_M=Nt-1)

        return check_save, n_gpucap
    
    def save_check(self, state_fwd_init, Nt, rec=None, model=None, v=None, tau=None, n_gpucap=24, **kwargs):
        '''
        Input
        state_fwd_init: doesn't need to be separated based on MPI size

        Output
        check_save: isn't combined when MPI is used
        '''
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        op_fwd = self.op_fwd()
        DTYPE = model.dtype
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])

        if (Nt-1)%n_gpucap ==0:
            n_gpucap = n_gpucap-1
            if (Nt-1)%n_gpucap ==0:
                num_check = (Nt-1)//n_gpucap
            else:
                num_check = (Nt-1)//n_gpucap+1
        else:
            num_check = (Nt-1)//n_gpucap+1
        # if kwargs.get('rank',0)==0:
        #     print('Number of checkpoints: {:d}'.format(num_check))
        _, Nxb, Nyb, Nzb = tau[0,0].data.shape
        check_save = np.zeros((num_check,9,Nxb,Nyb,Nzb),dtype=np.float32)
        self.state_assign(tau, v, 0, state_fwd_init, **kwargs)
        for cind in range(num_check):
            tindex = (cind*n_gpucap) % 3
            check_save[cind,0,:,:,:] = tau[0,0].data[tindex,:,:,:]
            check_save[cind,1,:,:,:] = tau[1,1].data[tindex,:,:,:]
            check_save[cind,2,:,:,:] = tau[2,2].data[tindex,:,:,:]
            check_save[cind,3,:,:,:] = tau[0,1].data[tindex,:,:,:]
            check_save[cind,4,:,:,:] = tau[1,2].data[tindex,:,:,:]
            check_save[cind,5,:,:,:] = tau[0,2].data[tindex,:,:,:]
            check_save[cind,6,:,:,:] = v[0].data[tindex,:,:,:]
            check_save[cind,7,:,:,:] = v[1].data[tindex,:,:,:]
            check_save[cind,8,:,:,:] = v[2].data[tindex,:,:,:]
            if cind!=(num_check-1):
                op_fwd.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                             sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                             sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], rec=rec,
                             time_m=cind*n_gpucap, time_M=(cind+1)*n_gpucap-1)
            else:
                op_fwd.apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                             sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                             sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], rec=rec,
                             time_m=cind*n_gpucap, time_M=Nt-1)

        return check_save, n_gpucap

    def sub_compute_gradient(self, state_fwd_init, state_bwd_init, measured_pressure_flat, v=None, tau=None, 
                             vsave=None, src=None, rec=None, model=None, grad_k_func=None, grad_m_func=None, **kwargs):
        """
        N/A
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        time = model.grid.time_dim
        s = time.spacing
        DTYPE = model.dtype
        nbl = model.nbl
        measured_pressure = measured_pressure_flat.reshape(-1,rec.data.shape[1])
        Nt_sub = measured_pressure.shape[0]
        savenum = Nt_sub
        
        # Setting new time function as initialization
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if vsave is None:
            vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, save=savenum)
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])

        if grad_k_func is None:
            grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        if grad_m_func is None:
            grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)

        # Setting initial pressure
        self.state_assign(tau, v, 0, state_fwd_init)
        self.op_jrfwd().apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                                sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                                sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], vsave_x=vsave[0],
                                vsave_y=vsave[1], vsave_z=vsave[2], rec=rec, time_m=0, time_M=savenum-1)

        src.data[:,:] = rec.data-measured_pressure

        tindex = (Nt_sub-1) % 3
        self.state_assign(tau, v, tindex, state_bwd_init)
        self.op_jradjcp().apply(dt=dt, v_adj_x=v[0], v_adj_y=v[1], v_adj_z=v[2], sigma_adj_xx=tau[0,0],
                            sigma_adj_xy=tau[0,1], sigma_adj_xz=tau[0,2], sigma_adj_yy=tau[1,1],
                            sigma_adj_yx=tau[1,0], sigma_adj_zx=tau[2,0], sigma_adj_zy=tau[2,1],
                            sigma_adj_yz=tau[1,2], sigma_adj_zz=tau[2,2], vsave_x=vsave[0],
                            vsave_y=vsave[1], vsave_z=vsave[2], grad_k=grad_k_func,
                            grad_m=grad_m_func, src=src, time_m=0, time_M=savenum-1)
        
        grad_p0 = -(tau[0,0].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau[1,1].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl]
                    +tau[2,2].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl])/3
        grad_k = grad_k_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()
        grad_m = grad_m_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()

        return rec.data.ravel(), grad_p0, grad_k, grad_m

    def sub_compute_gradient_mpi(self, state_fwd_init, state_bwd_init, measured_pressure_flat, Nt_sub, comm,
                                 v=None, tau=None, vsave=None, src=None, rec=None, model=None, grad_k_func=None, grad_m_func=None, terminate=False, **kwargs):
        '''
        Input
        state_fwd_init: doesn't need to be separated based on MPI size
        measured_pressure_flat: separated based on MPI size

        Intermediate
        measured_pressure: not separated based on MPI yet
        '''
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        time = model.grid.time_dim
        s = time.spacing
        DTYPE = model.dtype
        nbl = model.nbl
        Nx, Ny, Nz = model.shape
        Nxb = Nx + nbl*2
        Nyb = Ny + nbl*2
        Nzb = Nz + nbl*2
        rank = comm.Get_rank()
        size = comm.Get_size()

        measured_pressure = measured_pressure_flat.reshape(Nt_sub,-1)
        savenum = Nt_sub

        # Setting new time function as initialization
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if vsave is None:
            vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, save=savenum)
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])

        if grad_k_func is None:
            grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        if grad_m_func is None:
            grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        
        # Setting initial pressure
        self.state_assign(tau, v, 0, state_fwd_init)
        self.op_jrfwd().apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                                sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                                sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], vsave_x=vsave[0],
                                vsave_y=vsave[1], vsave_z=vsave[2], rec=rec, time_m=0, time_M=savenum-1)
        
        if size>1:
            p0_forward_T = rec.data[:,:].T.ravel()
            newData = comm.gather(p0_forward_T,root=0)
            if rank==0:
                p0_forward_T = np.array(newData,dtype=np.float32).ravel()
            else:
                p0_forward_T = np.zeros(p0_forward_T.shape[0]*size,dtype=np.float32)
            del newData
            comm.Bcast(p0_forward_T, root=0)
            p0_forward = p0_forward_T.reshape(-1,Nt_sub).T
        else:
            p0_forward = np.zeros(rec.data[:,:].shape,dtype=np.float32)
            p0_forward[:,:] = rec.data[:,:]

        src.data[:,:] = p0_forward-measured_pressure
        # src.data[:,:] = p0_forward

        tindex = (Nt_sub-1) % 3
        self.state_assign(tau, v, tindex, state_bwd_init)
        self.op_jradjcp().apply(dt=dt, v_adj_x=v[0], v_adj_y=v[1], v_adj_z=v[2], sigma_adj_xx=tau[0,0],
                            sigma_adj_xy=tau[0,1], sigma_adj_xz=tau[0,2], sigma_adj_yy=tau[1,1],
                            sigma_adj_yx=tau[1,0], sigma_adj_zx=tau[2,0], sigma_adj_zy=tau[2,1],
                            sigma_adj_yz=tau[1,2], sigma_adj_zz=tau[2,2], vsave_x=vsave[0],
                            vsave_y=vsave[1], vsave_z=vsave[2], grad_k=grad_k_func,
                            grad_m=grad_m_func, src=src, time_m=0, time_M=savenum-1)
        
        if terminate==True:
            p0_combine = -(tau[0,0].data[2,:,:,:]+tau[1,1].data[2,:,:,:]+tau[2,2].data[2,:,:,:])/3
            grad_p0_data = self.devito_mpi_collect(p0_combine, comm, Nxb, Nyb, Nzb)
            grad_p0 = grad_p0_data[nbl:-nbl,nbl:-nbl,nbl:-nbl]
            grad_k_data = self.devito_mpi_collect(grad_k_func.data, comm, Nxb, Nyb, Nzb)
            grad_k = grad_k_data[nbl:-nbl,nbl:-nbl,nbl:-nbl]
            grad_m_data = self.devito_mpi_collect(grad_m_func.data, comm, Nxb, Nyb, Nzb)
            grad_m = grad_m_data[nbl:-nbl,nbl:-nbl,nbl:-nbl]
            return p0_forward, grad_p0, grad_k, grad_m
        else:
            return 0,0,0,0
    
    def compute_gradient_selfcheck(self, sigma_v_p0, measured_pressure_flat, model=None, v=None, tau=None, vsave=None,
                                   grad_k_func=None, grad_m_func=None, rec=None, n_gpucap=24, **kwargs):
        """
        Not compatible with n_gpucap<=2 yet
        """
        if n_gpucap<=2:
            p0_forward, grad_p0, grad_k, grad_m = self.compute_gradient_check(sigma_v_p0,measured_pressure_flat.ravel(),checkpoint=True)
            return rec.data.ravel(), grad_p0, grad_k, grad_m

        Nt = self.rec.data.shape[0]
        model = model or self.model
        rec = rec or self.rec
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        DTYPE = model.dtype
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        if grad_k_func is None:
            grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        if grad_m_func is None:
            grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        check_save, n_gpucap = self.save_check(sigma_v_p0, Nt, v=v, tau=tau, n_gpucap=n_gpucap)
        n_check = check_save.shape[0]
        if vsave is None:
            vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, save=n_gpucap)

        measured_pressure = measured_pressure_flat.reshape(self.rec.data.shape[0],-1)        

        state_bwd_init = np.zeros(check_save.shape[1:],dtype=np.float32)
        for ind_check in np.arange(n_check,0,-1)-1:
            # print((ind_check+1)*n_gpucap, Nt-1)
            if (ind_check+1)*n_gpucap == Nt-1:
                measured_select = measured_pressure[ind_check*n_gpucap:,:].copy()
            else:
                measured_select = measured_pressure[ind_check*n_gpucap:(ind_check+1)*n_gpucap,:].copy()
            sub_Nt = measured_select.shape[0]
            # print(sub_Nt)
            if 'sub_rec' in locals() and sub_rec.data.shape[0]==sub_Nt:
                pass
            else:
                sub_time_range = TimeAxis(start=0, num=sub_Nt, step=dt)
                sub_rec = Receiver(name="subrec", grid=model.grid, npoint=self.rec.npoint, time_range=sub_time_range, dtype=DTYPE)
                sub_src = PointSource(name='subsrc', grid=model.grid, npoint=self.rec.npoint, time_range=sub_time_range, dtype=DTYPE)
                sub_rec.coordinates.data[:,:] = self.rec.coordinates.data[:,:]
                sub_src.coordinates.data[:,:] = self.src.coordinates.data[:,:]

            # if ind_check!=n_check-1:
            #     print(ind_check*n_gpucap,(ind_check+1)*n_gpucap-1)
            # else:
            #     print(ind_check*n_gpucap,Nt-1)
            state_fwd_init = check_save[ind_check,:,:,:,:]
            # input("Press Enter to continue... (before)")
            if sub_Nt==n_gpucap:
                _, grad_p0, grad_k, grad_m = self.sub_compute_gradient(state_fwd_init, state_bwd_init, measured_select.ravel(), v=v, tau=tau, grad_m_func=grad_m_func, vsave=vsave, grad_k_func=grad_k_func, rec=sub_rec, src=sub_src)
            else:
                _, grad_p0, grad_k, grad_m = self.sub_compute_gradient(state_fwd_init, state_bwd_init, measured_select.ravel(), v=v, tau=tau, grad_m_func=grad_m_func, grad_k_func=grad_k_func, rec=sub_rec, src=sub_src)
            # input("Press Enter to continue... (after)")
            state_bwd_init = self.state_extract(tau, v, 2)
            
        
        return rec.data.ravel(), grad_p0, grad_k, grad_m

    def compute_gradient_selfcheck_mpi(self, sigma_v_p0, measured_pressure_flat, rec_pos, model=None, v=None, tau=None, 
                                       vsave=None, grad_k_func=None, grad_m_func=None, rec=None, n_gpucap=24, **kwargs):
        """
        Not compatible with n_gpucap<=2 yet
        """
        assert(n_gpucap>2)
        Nt = self.rec.data.shape[0]
        model = model or self.model
        rec = rec or self.rec
        num_rec = rec_pos.shape[0]
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        DTYPE = model.dtype
        Nx, Ny, Nz = model.shape
        nbl = model.nbl
        Nxb = Nx + nbl*2
        Nyb = Ny + nbl*2
        Nzb = Nz + nbl*2
        comm = kwargs.get('comm', None)
        if comm is None:
            rank = 0
            size = 0
        else:
            rank = comm.Get_rank()
            size = comm.Get_size()
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        if grad_k_func is None:
            grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        if grad_m_func is None:
            grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        check_save, n_gpucap = self.save_check(sigma_v_p0, Nt, v=v, tau=tau, n_gpucap=n_gpucap, rank=rank)
        n_check = check_save.shape[0]
        if vsave is None:
            vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, save=n_gpucap)
        
        p0_forward_T = rec.data[:,:].T.ravel()
        if size>1:
            newData = comm.gather(p0_forward_T,root=0)
            if rank==0:
                p0_forward_T = np.array(newData,dtype=np.float32).ravel()
            else:
                p0_forward_T = np.zeros(p0_forward_T.shape[0]*size,dtype=np.float32)
            del newData
            comm.Bcast(p0_forward_T, root=0)
            p0_forward = p0_forward_T.reshape(-1,Nt).T
        else:
            p0_forward = p0_forward_T.reshape(-1,Nt).T

        measured_pressure = measured_pressure_flat.reshape(Nt,-1)

        state_bwd_init = np.zeros((9,Nxb,Nyb,Nzb),dtype=np.float32)
        # state_bwd_init = sigma_v_p0
        for ind_check in np.arange(n_check,0,-1)-1:
            if (ind_check+1)*n_gpucap == Nt-1:
                measured_select = measured_pressure[ind_check*n_gpucap:,:].copy()
            else:
                measured_select = measured_pressure[ind_check*n_gpucap:(ind_check+1)*n_gpucap,:].copy()
            sub_Nt = measured_select.shape[0]

            # sub_time_range = TimeAxis(start=0, num=sub_Nt, step=dt)
            # sub_rec = Receiver(name="subrec", grid=model.grid, npoint=num_rec, time_range=sub_time_range, dtype=DTYPE)
            # sub_src = PointSource(name='subsrc', grid=model.grid, npoint=num_rec, time_range=sub_time_range, dtype=DTYPE)
            # sub_rec.coordinates.data[:,:] = rec_pos
            # sub_src.coordinates.data[:,:] = rec_pos

            if 'sub_rec' in locals() and sub_rec.data.shape[0]==sub_Nt:
                pass
            else:
                sub_time_range = TimeAxis(start=0, num=sub_Nt, step=dt)
                sub_rec = Receiver(name="subrec", grid=model.grid, npoint=num_rec, time_range=sub_time_range, dtype=DTYPE)
                sub_src = PointSource(name='subsrc', grid=model.grid, npoint=num_rec, time_range=sub_time_range, dtype=DTYPE)
                sub_rec.coordinates.data[:,:] = rec_pos
                sub_src.coordinates.data[:,:] = rec_pos

            # if ind_check!=n_check-1 and rank==0:
            #     print(ind_check*n_gpucap,(ind_check+1)*n_gpucap-1)
            # elif rank==0:
            #     print(ind_check*n_gpucap,Nt-1)
            state_fwd_init = self.devito_mpi_collect(check_save[ind_check,:,:,:,:], comm, Nxb, Nyb, Nzb)
            terminate = ind_check==0
            if sub_Nt==n_gpucap:
                _, grad_p0, grad_k, grad_m = self.sub_compute_gradient_mpi(state_fwd_init, state_bwd_init, measured_select.ravel(), Nt_sub=sub_Nt, v=v, tau=tau, vsave=vsave, grad_m_func=grad_m_func, grad_k_func=grad_k_func, rec=sub_rec, src=sub_src, terminate=terminate, comm=comm)
            else:
                _, grad_p0, grad_k, grad_m = self.sub_compute_gradient_mpi(state_fwd_init, state_bwd_init, measured_select.ravel(), Nt_sub=sub_Nt, v=v, tau=tau, grad_m_func=grad_m_func, grad_k_func=grad_k_func, rec=sub_rec, src=sub_src, terminate=terminate, comm=comm)
            state_bwd_init = self.devito_mpi_collect(self.state_extract(tau, v, 2), comm, Nxb, Nyb, Nzb)
        return p0_forward, grad_p0, grad_k, grad_m
    
    def compute_gradient_check(self, sigma_and_v, rec_data_flat, v=None, tau=None, vsave=None, src=None, rec=None, model=None, checkpoint=False,n_checkpoints=None,**kwargs):
        """
        N/A
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        time = model.grid.time_dim
        s = time.spacing
        DTYPE = model.dtype
        nbl = model.nbl
        opt_roi = model.opt_roi
        (Nx,Ny,Nz) = model.shape
        savenum = self.time_range.num
        
        # Setting new time function as initialization
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
            if not checkpoint:
                vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, save=savenum)
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        v_adj = VectorTimeFunction(name='v_adj', grid=model.grid, space_order=self.space_order,
                               time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau_adj = TensorTimeFunction(name='sigma_adj', grid=model.grid, space_order=self.space_order, 
                                 time_order=self.time_order, dtype=DTYPE,
                                 staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)

        # Setting initial pressure
        tau[0,0].data[0,:,:,:] = sigma_and_v[0,:,:,:]
        tau[1,1].data[0,:,:,:] = sigma_and_v[1,:,:,:]
        tau[2,2].data[0,:,:,:] = sigma_and_v[2,:,:,:]
        tau[0,1].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,0].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,2].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[2,1].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[0,2].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        tau[2,0].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        v[0].data[0,:,:,:] = sigma_and_v[6,:,:,:]
        v[1].data[0,:,:,:] = sigma_and_v[7,:,:,:]
        v[2].data[0,:,:,:] = sigma_and_v[8,:,:,:]


        if checkpoint:
            cp = DevitoCheckpoint([v[0], v[1], v[2], tau[0,0], tau[1,1], tau[2,2], tau[0,1], tau[0,2], tau[1,2], tau[1,0], tau[2,0], tau[2,1]])
            wrap_fw = CheckpointOperator(self.op_fwd(), dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],
                                         sigma_xy=tau[0,1], sigma_xz=tau[0,2], sigma_yy=tau[1,1],
                                         sigma_yx=tau[1,0], sigma_zx=tau[2,0], sigma_zy=tau[2,1],
                                         sigma_yz=tau[1,2], sigma_zz=tau[2,2], rec=rec)
            wrap_adj = CheckpointOperator(self.op_jradjcp(checkpoint), dt=dt, v_adj_x=v_adj[0], v_adj_y=v_adj[1],
                                          v_adj_z=v_adj[2], sigma_adj_xx=tau_adj[0,0], sigma_adj_xy=tau_adj[0,1],
                                          sigma_adj_xz=tau_adj[0,2], sigma_adj_yy=tau_adj[1,1], sigma_adj_yx=tau_adj[1,0],
                                          sigma_adj_zx=tau_adj[2,0], sigma_adj_zy=tau_adj[2,1], sigma_adj_yz=tau_adj[1,2],
                                          sigma_adj_zz=tau_adj[2,2], vsave_x=v[0], vsave_y=v[1], vsave_z=v[2],
                                          grad_k=grad_k_func, grad_m=grad_m_func, src=src)
            wrp = Revolver(cp, wrap_fw, wrap_adj, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()

            measured_pressure = rec_data_flat.reshape(rec.data.shape[0],-1)
            src.data[:,:] = rec.data-measured_pressure

            wrp.apply_reverse()
        else:
            self.op_jrfwd().apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                                  sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                                  sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], vsave_x=vsave[0],
                                  vsave_y=vsave[1], vsave_z=vsave[2], rec=rec, time_m=0, time_M=self.time_range.num-1)
            measured_pressure = rec_data_flat.reshape(rec.data.shape[0],-1)
            src.data[:,:] = rec.data-measured_pressure
            self.op_jradjcp(checkpoint).apply(dt=dt, v_adj_x=v_adj[0], v_adj_y=v_adj[1], v_adj_z=v_adj[2], sigma_adj_xx=tau_adj[0,0],
                                sigma_adj_xy=tau_adj[0,1], sigma_adj_xz=tau_adj[0,2], sigma_adj_yy=tau_adj[1,1],
                                sigma_adj_yx=tau_adj[1,0], sigma_adj_zx=tau_adj[2,0], sigma_adj_zy=tau_adj[2,1],
                                sigma_adj_yz=tau_adj[1,2], sigma_adj_zz=tau_adj[2,2], vsave_x=vsave[0],
                                vsave_y=vsave[1], vsave_z=vsave[2], grad_k=grad_k_func,
                                grad_m=grad_m_func, src=src, time_m=0, time_M=self.time_range.num-1)
        
        grad_p0 = -(tau_adj[0,0].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau_adj[1,1].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl]
                    +tau_adj[2,2].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl])/3
        grad_k = grad_k_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()
        grad_m = grad_m_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()

        return rec.data.ravel(), grad_p0, grad_k, grad_m

    def compute_gradient_p0(self, sigma_and_v, rec_data_flat, v=None, tau=None, src=None, rec=None, model=None,**kwargs):
        """
        N/A
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        time = model.grid.time_dim
        DTYPE = model.dtype
        nbl = model.nbl
        
        # Setting new time function as initialization
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        v_adj = VectorTimeFunction(name='v_adj', grid=model.grid, space_order=self.space_order,
                                time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau_adj = TensorTimeFunction(name='sigma_adj', grid=model.grid, space_order=self.space_order, 
                                    time_order=self.time_order, dtype=DTYPE,
                                    staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])

        # Setting initial pressure
        tau[0,0].data[0,:,:,:] = sigma_and_v[0,:,:,:]
        tau[1,1].data[0,:,:,:] = sigma_and_v[1,:,:,:]
        tau[2,2].data[0,:,:,:] = sigma_and_v[2,:,:,:]
        tau[0,1].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,0].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,2].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[2,1].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[0,2].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        tau[2,0].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        v[0].data[0,:,:,:] = sigma_and_v[6,:,:,:]
        v[1].data[0,:,:,:] = sigma_and_v[7,:,:,:]
        v[2].data[0,:,:,:] = sigma_and_v[8,:,:,:]

        self.op_fwd().apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],sigma_xy=tau[0,1],
                            sigma_xz=tau[0,2], sigma_yy=tau[1,1],sigma_yx=tau[1,0], sigma_zx=tau[2,0],
                            sigma_zy=tau[2,1],sigma_yz=tau[1,2], sigma_zz=tau[2,2], rec=rec, time_m=0, time_M=self.time_range.num-1)
        measured_pressure = rec_data_flat.reshape(rec.data.shape[0],-1)
        src.data[:,:] = rec.data-measured_pressure
        self.op_adj().apply(dt=dt, v_x=v_adj[0], v_y=v_adj[1], v_z=v_adj[2], sigma_xx=tau_adj[0,0],
                            sigma_xy=tau_adj[0,1], sigma_xz=tau_adj[0,2], sigma_yy=tau_adj[1,1],
                            sigma_yx=tau_adj[1,0], sigma_zx=tau_adj[2,0], sigma_zy=tau_adj[2,1],
                            sigma_yz=tau_adj[1,2], sigma_zz=tau_adj[2,2], src=src, time_m=0, time_M=self.time_range.num-1)
        
        grad_p0 = -(tau_adj[0,0].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau_adj[1,1].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl]
                    +tau_adj[2,2].data[2,nbl:-nbl,nbl:-nbl,nbl:-nbl])/3

        return rec.data.ravel(), grad_p0

    def compute_gradient_legacy_2(self, sigma_and_v, rec_data_flat, v=None, tau=None, vsave=None, src=None, rec=None, model=None, **kwargs):
        """
        Outdated 05.01.2024
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        time = model.grid.time_dim
        s = time.spacing
        DTYPE = model.dtype
        nbl = model.nbl
        opt_roi = model.opt_roi
        (Nx,Ny,Nz) = model.shape
        savenum = self.time_range.num
        
        # Setting new time function as initialization
        if v is None:
            v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        if tau is None:
            tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        if vsave is None:
            vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order, time_order=self.time_order, dtype=DTYPE, save=savenum)
        
        # Setting initial pressure
        tau[0,0].data[0,:,:,:] = sigma_and_v[0,:,:,:]
        tau[1,1].data[0,:,:,:] = sigma_and_v[1,:,:,:]
        tau[2,2].data[0,:,:,:] = sigma_and_v[2,:,:,:]
        tau[0,1].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,0].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,2].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[2,1].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[0,2].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        tau[2,0].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        v[0].data[0,:,:,:] = sigma_and_v[6,:,:,:]
        v[1].data[0,:,:,:] = sigma_and_v[7,:,:,:]
        v[2].data[0,:,:,:] = sigma_and_v[8,:,:,:]
        
        self.op_jrfwd().apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],
                     sigma_xy=tau[0,1], sigma_xz=tau[0,2], sigma_yy=tau[1,1],
                     sigma_yx=tau[1,0], sigma_zx=tau[2,0], sigma_zy=tau[2,1],
                     sigma_yz=tau[1,2], sigma_zz=tau[2,2], vsave_x=vsave[0],
                     vsave_y=vsave[1], vsave_z=vsave[2], rec=rec,
                     time_m=0, time_M=self.time_range.num-1)
        
        v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order,
                               time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, 
                                 time_order=self.time_order, dtype=DTYPE,
                                 staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        
        measured_pressure = rec_data_flat.reshape(rec.data.shape[0],-1)
        src.data[:,:] = rec.data-measured_pressure
        
        self.op_jradj().apply(dt=dt, v_x=v[0], v_y=v[1], v_z=v[2], sigma_xx=tau[0,0],
                     sigma_xy=tau[0,1], sigma_xz=tau[0,2], sigma_yy=tau[1,1],
                     sigma_yx=tau[1,0], sigma_zx=tau[2,0], sigma_zy=tau[2,1],
                     sigma_yz=tau[1,2], sigma_zz=tau[2,2], vsave_x=vsave[0],
                     vsave_y=vsave[1], vsave_z=vsave[2], grad_k=grad_k_func,
                     grad_m=grad_m_func, src=src, time_m=0, time_M=self.time_range.num-1)
        
        grad_p0 = -(tau[0,0].data[0,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau[1,1].data[0,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau[2,2].data[0,nbl:-nbl,nbl:-nbl,nbl:-nbl])/3
        grad_k = grad_k_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()
        grad_m = grad_m_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()

        return rec.data.ravel(), grad_p0, grad_k, grad_m
    
    def compute_gradient_legacy(self, sigma_and_v, rec_data_flat, src=None, rec=None, model=None, save=None, **kwargs):
        """
        Outdated 04.15.2024
        """
        # Source term is read-only, so re-use the default
        src = src or self.src
        rec = rec or self.rec
        model = model or self.model
        dt = self.time_range.step
        x, y, z = model.grid.dimensions
        t = model.grid.stepping_dim
        time = model.grid.time_dim
        s = time.spacing
        DTYPE = model.dtype
        nbl = model.nbl
        opt_roi = model.opt_roi
        (Nx,Ny,Nz) = model.shape
        savenum = self.time_range.num
        
        # Setting new time function as initialization
        v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order,
                           time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, 
                             time_order=self.time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=self.space_order,
                                   time_order=self.time_order, dtype=DTYPE, save=savenum)
        # tausave = TensorTimeFunction(name='sigmasave', grid=model.grid, space_order=self.space_order,
        #                              time_order=self.time_order, dtype=DTYPE,save=savenum)
        
        # Setting initial pressure
        tau[0,0].data[0,:,:,:] = sigma_and_v[0,:,:,:]
        tau[1,1].data[0,:,:,:] = sigma_and_v[1,:,:,:]
        tau[2,2].data[0,:,:,:] = sigma_and_v[2,:,:,:]
        tau[0,1].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,0].data[0,:,:,:] = sigma_and_v[3,:,:,:]
        tau[1,2].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[2,1].data[0,:,:,:] = sigma_and_v[4,:,:,:]
        tau[0,2].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        tau[2,0].data[0,:,:,:] = sigma_and_v[5,:,:,:]
        v[0].data[0,:,:,:] = sigma_and_v[6,:,:,:]
        v[1].data[0,:,:,:] = sigma_and_v[7,:,:,:]
        v[2].data[0,:,:,:] = sigma_and_v[8,:,:,:]
        
        rec_term = rec.interpolate(expr=-(tau[0, 0] + tau[1, 1] + tau[2, 2]) / 3)
        
        # Split coordinate pml
        u_vx = Eq(v.forward[0], v[0] * (1 - s*model.s_x - s*model.alp_x) + s*model.b_x*div(tau)[0])
        u_vy = Eq(v.forward[1], v[1] * (1 - s*model.s_y - s*model.alp_y) + s*model.b_y*div(tau)[1])
        u_vz = Eq(v.forward[2], v[2] * (1 - s*model.s_z - s*model.alp_z) + s*model.b_z*div(tau)[2])

        u_txx = Eq(tau.forward[0, 0], (tau[0, 0] * (1 - s*model.s_x) + s* (model.lam_diag*diag(div(v.forward)) +
                            model.mu_diag * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[0, 0]))
        u_tyy = Eq(tau.forward[1, 1], (tau[1, 1] * (1 - s*model.s_y) + s* (model.lam_diag*diag(div(v.forward)) +
                            model.mu_diag * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[1, 1]))
        u_tzz = Eq(tau.forward[2, 2], (tau[2, 2] * (1 - s*model.s_z) + s* (model.lam_diag*diag(div(v.forward)) +
                            model.mu_diag * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[2, 2]))

        u_tyx = Eq(tau.forward[1, 0], (tau[1, 0]*(1-s*model.s_y) + s* (model.mu_xy * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[1, 0]))
        u_txy = Eq(tau.forward[0, 1], (tau[0, 1]*(1-s*model.s_x) + s* (model.mu_xy * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[0, 1]))
        u_tzx = Eq(tau.forward[2, 0], (tau[2, 0]*(1-s*model.s_z) + s* (model.mu_xz * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[2, 0]))
        u_txz = Eq(tau.forward[0, 2], (tau[0, 2]*(1-s*model.s_x) + s* (model.mu_xz * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[0, 2]))
        u_tyz = Eq(tau.forward[1, 2], (tau[1, 2]*(1-s*model.s_y) + s* (model.mu_yz * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[1, 2]))
        u_tzy = Eq(tau.forward[2, 1], (tau[2, 1]*(1-s*model.s_z) + s* (model.mu_yz * (grad(v.forward)+grad(v.forward).transpose(inner=False)))[2, 1]))

        # op_fwd = Operator([u_vx, u_vy, u_vz]+[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz]
        #                   +rec_term+[Eq(vsave, v)]+[Eq(tausave, tau)],
        #                   compiler='nvc', platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))
        op_fwd = Operator([u_vx, u_vy, u_vz]+[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz]
                          +rec_term+[Eq(vsave, v)],
                          compiler='nvc', platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))
        op_fwd.apply(dt=dt, time_m=0, time_M=self.time_range.num-1)
        
        v = VectorTimeFunction(name='v', grid=model.grid, space_order=self.space_order,
                               time_order=self.time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
        tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=self.space_order, 
                                 time_order=self.time_order, dtype=DTYPE,
                                 staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
        grad_k_func = Function(name='grad_k', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        grad_m_func = Function(name='grad_m', grid=model.grid, space_order=self.space_order, dtype=DTYPE)
        
        measured_pressure = rec_data_flat.reshape(rec.data.shape[0],-1)
        src.data[:,:] = rec.data-measured_pressure
        src_xx = src.inject(field=tau.backward[0, 0], expr=-src)
        src_yy = src.inject(field=tau.backward[1, 1], expr=-src)
        src_zz = src.inject(field=tau.backward[2, 2], expr=-src)
        
        # Split coordinate pml
        u_vx = Eq(v.backward[0], v[0] * (1 - s*model.s_x - s*model.alp_x) + s*model.b_x*div(tau)[0])
        u_vy = Eq(v.backward[1], v[1] * (1 - s*model.s_y - s*model.alp_y) + s*model.b_y*div(tau)[1])
        u_vz = Eq(v.backward[2], v[2] * (1 - s*model.s_z - s*model.alp_z) + s*model.b_z*div(tau)[2])

        u_txx = Eq(tau.backward[0, 0], (tau[0, 0] * (1 - s*model.s_x) 
                                        + s* (model.lam_diag*diag(div(v.backward)) 
                                              + model.mu_diag * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[0, 0]))
        u_tyy = Eq(tau.backward[1, 1], (tau[1, 1] * (1 - s*model.s_y)
                                        + s* (model.lam_diag*diag(div(v.backward)) 
                                              + model.mu_diag * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[1, 1]))
        u_tzz = Eq(tau.backward[2, 2], (tau[2, 2] * (1 - s*model.s_z) 
                                        + s* (model.lam_diag*diag(div(v.backward)) 
                                              + model.mu_diag * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[2, 2]))

        u_tyx = Eq(tau.backward[1, 0], (tau[1, 0]*(1-s*model.s_y) +
                                        s*(model.mu_xy * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[1, 0]))
        u_txy = Eq(tau.backward[0, 1], (tau[0, 1]*(1-s*model.s_x) +
                                        s*(model.mu_xy * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[0, 1]))
        u_tzx = Eq(tau.backward[2, 0], (tau[2, 0]*(1-s*model.s_z) +
                                        s*(model.mu_xz * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[2, 0]))
        u_txz = Eq(tau.backward[0, 2], (tau[0, 2]*(1-s*model.s_x) +
                                        s*(model.mu_xz * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[0, 2]))
        u_tyz = Eq(tau.backward[1, 2], (tau[1, 2]*(1-s*model.s_y) +
                                        s*(model.mu_yz * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[1, 2]))
        u_tzy = Eq(tau.backward[2, 1], (tau[2, 1]*(1-s*model.s_z) +
                                        s*(model.mu_yz * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[2, 1]))
        
        grad_ksum = Eq(grad_k_func, (grad_k_func + (grad(vsave)[0,0]+grad(vsave)[1,1]+grad(vsave)[2,2]) 
                                     * (tau[0, 0]+tau[1, 1]+tau[2, 2])))
        grad_msum = Eq(grad_m_func, (grad_m_func + (4*tau[0,0]/3-2*tau[1,1]/3-2*tau[2,2]/3)*grad(vsave)[0,0]
                                     + (4*tau[1,1]/3-2*tau[0,0]/3-2*tau[2,2]/3)*grad(vsave)[1,1]
                                     + (4*tau[2,2]/3-2*tau[0,0]/3-2*tau[1,1]/3)*grad(vsave)[2,2]
                                     + (tau[0,1]+tau[1,0])*(grad(vsave)[0,1]+grad(vsave)[1,0])
                                     + (tau[1,2]+tau[2,1])*(grad(vsave)[1,2]+grad(vsave)[2,1])
                                     + (tau[2,0]+tau[0,2])*(grad(vsave)[2,0]+grad(vsave)[0,2])))
        op = Operator([u_vx, u_vy, u_vz]
                      +[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy,u_tzz]
                      +src_xx+src_yy+src_zz+[grad_ksum]+[grad_msum],
                      compiler='nvc',platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))
        
        op.apply(dt=dt, time_m=0, time_M=self.time_range.num-1)
        
        grad_p0 = -(tau[0,0].data[0,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau[1,1].data[0,nbl:-nbl,nbl:-nbl,nbl:-nbl]+tau[2,2].data[0,nbl:-nbl,nbl:-nbl,nbl:-nbl])/3
        grad_k = grad_k_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()
        grad_m = grad_m_func.data[nbl:-nbl,nbl:-nbl,nbl:-nbl].copy()

        return rec.data.ravel(), grad_p0, grad_k, grad_m
