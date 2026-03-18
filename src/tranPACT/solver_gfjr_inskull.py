import numpy as np
from .solver_basic import TranPACTWaveSolver
from .solver_optim import OptimParam
import os
import pybobyqa

__all__ = ['GFJRISSolver']

class ReducedCostFunction:
    """
    Calculates the reduced cost function for the optimization.
    This class is used to compute the cost function for a given set of medium parameters.
    It uses the FISTA-TV algorithm to solve the inverse problem for a given set of medium
    parameters at each GFJR iteration.
    The cost function is defined as the L2 norm of the difference between the measured
    pressure and the simulated pressure.
    Note that the medium_set function should be updated for the specific problem.
    """
    def __init__(self, p0_est, measured_pressure, fn, opt_param, wave_solver, 
                 forward, adjoint, **param):
        """
        Initializes the ReducedCostFunction.

        Args:
            p0_est: np.ndarray
                The 3D initial estimate of the pressure field.
            measured_pressure: np.ndarray
                The observed pressure data (2D or 1D)
            fn: np.ndarray
                The medium parameters (2D, num_mediums x 4).
            opt_param: object
                An object containing optimization parameters, including:
                - out_print: Flag for printing output.
                - reg: Regularization parameter.
                - lip: Lipschitz constant.
                - positive_constraint: Flag for positivity constraint.
                - num_iter: Maximum number of iterations.
            wave_solver: TranPACTWaveSolver
                An instance of the TranPACTWaveSolver class, used to perform
                forward and adjoint modeling.
            saving_dir: str
                The directory where intermediate and final results are saved.
            forward: callable
                The forward modeling function.
            adjoint: callable
                The adjoint modeling function.
            **param: dict, optional
                Additional parameters, including:
                - rank: The MPI rank (default: 0).
                - size: The MPI size (default: 1).
                - comm: The MPI communicator (default: None).
                - Nt: The number of time steps (default: 4800).
                - use_check: Flag to use existing data from a previous run (default: False).
        """
        ## General parameters
        import fista_tv_overall
        self.p0_est = p0_est
        self.p0_est_global = self.p0_est.copy()
        self.measured_pressure = measured_pressure
        self.fn = fn
        self.opt_para = opt_param
        self.saving_dir = opt_param.saving_dir
        self.solver = wave_solver
        (Nx, Ny, Nz) = self.solver.model.shape
        self.forward = forward
        self.adjoint = adjoint

        self.inner_solver = lambda init_guess :fista_tv_overall.fista_tv(self.measured_pressure.ravel(), Nx, Ny, Nz, self.forward, self.adjoint, self.solver.model.opt_roi, out_print=self.opt_para.out_print, reg_param=self.opt_para.reg, lip=self.opt_para.lip, positive_constraint=self.opt_para.positive_constraint, niter_max=self.opt_para.num_iter, grad_min=1e-5, cost_min=1e-3, saving_dir=self.saving_dir, use_check=False, check_iter=10, init_guess=init_guess, save_freq=1, mpi_rank=self.rank)

        self.truep0 = param.get("truep0", None)
        self.optroi = param.get("optroi", None)
        self.medium_mode = wave_solver.model.medium_mode

        ## MPI parameters
        self.rank = param.get("rank", 0)
        ## Load parameters
        use_check = opt_param.use_check
        self.cur_iter = 0
        if use_check and os.path.exists(self.saving_dir+'datafid_record.DAT'):
            print("load existing GFJR data...")
            self.obj_record = np.fromfile(self.saving_dir+"datafid_record.DAT", dtype=np.float32).tolist()
            self.iter_record = np.fromfile(self.saving_dir+"iter_record.DAT", dtype=np.float32).tolist()
            self.c0_record = np.fromfile(self.saving_dir+"c0_record.DAT", dtype=np.float32).reshape(len(self.obj_record), -1).tolist()
        else:
            print("start GFJR from scratch...")
            self.c0_record = []
            self.obj_record = []
            self.iter_record = []
    
    def medium_set(self, sos_local):
        '''
        Setting the medium parameter in the model based on the input sos
        Args:
            sos_local: The local speed of sound (sos) values.
        '''
        if self.medium_mode == 'mpml':
            if (self.fn.shape[0] == 5):
                fn_tmp = self.fn.copy()
                fn_tmp[2,:] = self.fn[2,:]
                fn_tmp[0,0] = self.fn[0,0]
                fn_tmp[0,2] = sos_local[1]**2 * self.fn[0,0]
                fn_tmp[0,1] = sos_local[0]**2 * self.fn[0,0] - 4*fn_tmp[0,2]/3
                fn_tmp[3,0] = self.fn[3,0]
                fn_tmp[3,2] = sos_local[3]**2 * self.fn[3,0]
                fn_tmp[3,1] = sos_local[2]**2 * self.fn[3,0] - 4*fn_tmp[3,2]/3
                fn_tmp[[1,4],:] = fn_tmp[3,:]
                self.solver.model.initialize_medium(fn_tmp, water_index=2)
            elif (self.fn.shape[0] == 2):
                fn_tmp = self.fn.copy()
                fn_tmp[0,:] = self.fn[0,:]
                fn_tmp[1,0] = self.fn[1,0]
                fn_tmp[1,2] = sos_local[1]**2 * self.fn[1,0]
                fn_tmp[1,1] = sos_local[0]**2 * self.fn[1,0] - 4*fn_tmp[1,2]/3
                self.solver.model.initialize_medium(fn_tmp, water_index=0)
            elif (self.fn.shape[0] == 3):
                fn_tmp = self.fn.copy()
                fn_tmp[0,:] = self.fn[0,:]
                fn_tmp[1,0] = self.fn[1,0]
                fn_tmp[1,2] = sos_local[1]**2 * self.fn[1,0]
                fn_tmp[1,1] = sos_local[0]**2 * self.fn[1,0] - 4*fn_tmp[1,2]/3
                fn_tmp[2,0] = self.fn[2,0]
                fn_tmp[2,2] = sos_local[3]**2 * self.fn[2,0]
                fn_tmp[2,1] = sos_local[2]**2 * self.fn[2,0] - 4*fn_tmp[2,2]/3
                self.solver.model.initialize_medium(fn_tmp, water_index=0)
        elif self.medium_mode == 'aubry':
            if (self.fn.shape[0] != 2):
                raise ValueError("For aubry mode, fn should have shape (2, 4).")
            if sos_local.shape[0] == 2:
                fn_tmp = self.fn.copy()
                fn_tmp[0,:] = self.fn[0,:]
                fn_tmp[1,0] = self.fn[1,0]
                fn_tmp[1,2] = sos_local[1]**2 * self.fn[1,0]
                fn_tmp[1,1] = sos_local[0]**2 * self.fn[1,0] - 4*fn_tmp[1,2]/3
            elif sos_local.shape[0] == 4:
                fn_tmp = self.fn.copy()
                fn_tmp[0,:] = self.fn[0,:]
                fn_tmp[1,0] = sos_local[0]
                fn_tmp[1,2] = sos_local[2]**2 * self.fn[1,0]
                fn_tmp[1,1] = sos_local[1]**2 * self.fn[1,0] - 4*fn_tmp[1,2]/3
                fn_tmp[1,3] = sos_local[3]
            else:
                raise ValueError("sos_local should have shape (2,) or (4,).")
            self.solver.model.initialize_medium(fn_tmp)
        else:
            raise ValueError(f"Unknown medium mode: {self.medium_mode}. Supported modes are 'mpml' and 'aubry'.")

    
    def __call__(self, sos_local):
        """
        Computes the reduced cost function.
        The first part is to determine whether the cost function at the current itertation is computed or not.
        If it is computed, then load the cost function and return it.
        If it is not computed, the first step is to set the medium.
        The second step is to solve the inner loop.
        The third step is to compute the cost function and return it.
        Args:
            sos_local: The local speed of sound (sos) values.
        Returns:
            The computed cost function value.
        """
        ## Step 1: Check if the cost function at the current iteration is computed or not.
        iter_ind = len(self.obj_record)
        if self.cur_iter < iter_ind:
            if self.cur_iter == iter_ind-1:
                tarindex = int(np.argmin(self.obj_record)+1)
            cost = self.obj_record[self.cur_iter]
            self.cur_iter += 1
            if self.rank==0:
                print('Load computed GFJR: ',sos_local, '{:.4f}'.format(cost))
            return cost

        ## Step 2: Solve the inner loop.
        self.medium_set(sos_local)
        self.p0_est, _, fistaiter = self.inner_solver(self.p0_est_global)

        ## Step 3: Compute the cost function.
        if self.truep0 is not None:
            cost = np.linalg.norm((self.p0_est-self.truep0)[self.optroi>0])**2
        else:
            pest = self.forward(self.p0_est)
            cost = np.linalg.norm(self.measured_pressure.ravel() - pest)**2

        ## Step 4: Save the computed cost function and the current sos_local.
        self.c0_record.append(sos_local.tolist())
        self.obj_record.append(cost)
        self.iter_record.append(fistaiter)
        iter_ind = len(self.obj_record)
        if self.rank==0:
            try:
                print(sos_local, '{:.4f}'.format(cost), fistaiter)
                self.p0_est.astype(np.float32).tofile(self.saving_dir+f'gfjr_{iter_ind}.DAT')
                np.array(self.obj_record, dtype=np.float32).tofile(self.saving_dir+'datafid_record.DAT')
                np.array(self.c0_record, dtype=np.float32).tofile(self.saving_dir+'c0_record.DAT')
                np.array(self.iter_record, dtype=np.float32).tofile(self.saving_dir+'iter_record.DAT')
            except OSError as err:
                print(err)
            except Exception as e:
                print(e)
        self.cur_iter += 1
        return cost

class GFJRISSolver:
    """
    Joint Reconstruction Wave Solver using pybobyqa.
    """
    def __init__(self, solver, measured_pressure, fn, roi_inskull, forward=None, adjoint=None, opt_param=None, init_guess=None, **param):
        self.solver = solver
        self.data = measured_pressure.copy()
        self.opt_para = opt_param or OptimParam()
        if init_guess is not None:
            self.p0_est = init_guess
        else:
            self.p0_est = np.zeros(solver.model.shape, dtype=np.float32)
        self.roi_inskull = roi_inskull
        self.fn = fn

        if forward is None:
            self.forward = self.solver.forward
        else:
            self.forward = forward
        if adjoint is None:
            self.adjoint = self.solver.adjoint
        else:
            self.adjoint = adjoint
        self.saving_dir = opt_param.saving_dir

        self.comm = param.get("comm", None)
        
        self.rank = 0
        self.size = 1
        if self.comm is not None:
            Nt = param.get("Nt", 4800)
            (Nx, Ny, Nz) = solver.model.shape
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.forward = lambda p0: solver.mpi_forward(p0, comm=self.comm, Nt=Nt)
            self.adjoint = lambda p0_forward: solver.mpi_adjoint(p0_forward.copy(), comm=self.comm, Nx=Nx, Ny=Ny, Nz=Nz, fullscale=False)
        if self.rank == 0:
            if not os.path.exists(self.saving_dir):
                os.system('mkdir '+self.saving_dir)
        self.cost_function = ReducedCostFunction(p0_est=self.p0_est, measured_pressure=self.data, fn=self.fn, opt_param=self.opt_para, wave_solver=self.solver, forward=self.forward, adjoint=self.adjoint, rank=self.rank, **param)
        

    def solve(self, c0, lower, upper, num_iter_init=None):
        """
        Runs the optimization using pybobyqa to estimate the speed of sound (c).

        Returns:
            The estimated speed of sound (c) as a NumPy array.
        """
        if not os.path.exists(self.saving_dir+'OBRM_result.DAT'):
            self.cost_function.medium_set(c0)
            self.initial_guess(num_iter_init)
        (Nx, Ny, Nz) = self.solver.model.shape
        self.cost_function.p0_est_global = np.fromfile(self.saving_dir+'OBRM_result.DAT', dtype=np.float32).reshape(Nx,Ny,Nz)
        ## clean in skull area
        self.cost_function.p0_est_global[self.roi_inskull==1] = 0.0

        soln = pybobyqa.solve(self.cost_function, c0, rhobeg=0.2, rhoend=0.01, maxfun=60, scaling_within_bounds=True, bounds=(lower,upper))
        if self.rank==0:
            print(soln)

    def initial_guess(self, num_iter=None):
        import fista_tv_overall
        num_iter_init = num_iter or self.opt_para.num_iter
        OBRM_path = self.saving_dir + 'OBRM/'
        if not os.path.exists(OBRM_path) and self.rank==0:
            os.system('mkdir '+OBRM_path)
        (Nx, Ny, Nz) = self.solver.model.shape
        try:
            self.p0_est, _, _ = fista_tv_overall.fista_tv(self.data.ravel(), Nx, Ny, Nz, self.forward, self.adjoint, self.solver.model.opt_roi, out_print=self.opt_para.out_print, reg_param=self.opt_para.reg, lip=self.opt_para.lip, positive_constraint=self.opt_para.positive_constraint, niter_max=num_iter_init, grad_min=1e-4, cost_min=1e-3, saving_dir=OBRM_path, use_check=True, check_iter=10, init_guess=self.p0_est, save_freq=1, mpi_rank=self.rank)
            self.p0_est.astype(np.float32).tofile(self.saving_dir+f'OBRM_result.DAT')
        except OSError as err:
            print(err)

