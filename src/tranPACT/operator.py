import numpy as np
from devito import (VectorTimeFunction, TensorTimeFunction, Function, Eq,
                    div, Operator, grad, diag)

__all__ = ['ForwardOperator','AdjointOperator','JRForwardOperator','JRAdjointOperator','JRAdjointcheckOperator']

def ForwardOperator(model, p0, rec, space_order=10, time_order=2, **kwargs):
    x, y, z = model.grid.dimensions
    t = model.grid.stepping_dim
    time = model.grid.time_dim
    s = time.spacing
    DTYPE = model.dtype
    nbl = model.nbl
    opt_roi = model.opt_roi
    (Nx,Ny,Nz) = model.shape
    
    v = VectorTimeFunction(name='v', grid=model.grid, space_order=space_order,
                           time_order=time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])

    tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=space_order, 
                             time_order=time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
    
    p0_fullsize = np.zeros(model.shape)
    p0_fullsize[opt_roi>0] = p0
    tau[0,0].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0_fullsize
    tau[1,1].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0_fullsize
    tau[2,2].data[0,nbl:nbl+Nx,nbl:nbl+Ny,nbl:nbl+Nz] = -p0_fullsize
    
    if type(rec)==type([]):
        rec_term = rec[0].interpolate(expr=-(tau[0, 0] + tau[1, 1] + tau[2, 2]) / 3)
        for rind in range(len(rec)-1):
            rec_term += rec[rind+1].interpolate(expr=-(tau[0, 0] + tau[1, 1] + tau[2, 2]) / 3)
    else:
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

    return Operator([u_vx, u_vy, u_vz] + [u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz] + rec_term,
                    compiler='nvc', platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}), **kwargs)

def AdjointOperator(model, rec, src, space_order=10, time_order=2, **kwargs):
    x, y, z = model.grid.dimensions
    t = model.grid.stepping_dim
    time = model.grid.time_dim
    s = time.spacing
    DTYPE = model.dtype
    
    v = VectorTimeFunction(name='v', grid=model.grid, space_order=space_order,
                           time_order=time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])

    tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=space_order, 
                             time_order=time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])

    src_xx = src.inject(field=tau.backward[0, 0], expr=-src)
    src_yy = src.inject(field=tau.backward[1, 1], expr=-src)
    src_zz = src.inject(field=tau.backward[2, 2], expr=-src)
    
    # Split coordinate pml
    u_vx = Eq(v.backward[0], v[0] * (1 - s*model.s_x - s*model.alp_x) + s*model.b_x*div(tau)[0])
    u_vy = Eq(v.backward[1], v[1] * (1 - s*model.s_y - s*model.alp_y) + s*model.b_y*div(tau)[1])
    u_vz = Eq(v.backward[2], v[2] * (1 - s*model.s_z - s*model.alp_z) + s*model.b_z*div(tau)[2])

    u_txx = Eq(tau.backward[0, 0], (tau[0, 0] * (1 - s*model.s_x) + s* (model.lam_diag*diag(div(v.backward)) +
                        model.mu_diag * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[0, 0]))
    u_tyy = Eq(tau.backward[1, 1], (tau[1, 1] * (1 - s*model.s_y) + s* (model.lam_diag*diag(div(v.backward)) +
                        model.mu_diag * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[1, 1]))
    u_tzz = Eq(tau.backward[2, 2], (tau[2, 2] * (1 - s*model.s_z) + s* (model.lam_diag*diag(div(v.backward)) +
                        model.mu_diag * (grad(v.backward)+grad(v.backward).transpose(inner=False)))[2, 2]))

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

    return Operator([u_vx, u_vy, u_vz]+[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz]+src_xx+src_yy+src_zz,
                    compiler='nvc', platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}), **kwargs)

def JRForwardOperator(model, rec, space_order=10, time_order=2, **kwargs):
    x, y, z = model.grid.dimensions
    t = model.grid.stepping_dim
    time = model.grid.time_dim
    s = time.spacing
    DTYPE = model.dtype
    savenum = rec.data.shape[0]

    # Setting new time function as initialization
    v = VectorTimeFunction(name='v', grid=model.grid, space_order=space_order,
                       time_order=time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
    tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=space_order, 
                         time_order=time_order, dtype=DTYPE,
                         staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
    vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=space_order,
                               time_order=time_order, dtype=DTYPE, save=savenum)

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

    return Operator([u_vx, u_vy, u_vz]+[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz]+rec_term+[Eq(vsave, v)],
                    compiler='nvc', platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))

def JRAdjointOperator(model, rec, src, space_order=10, time_order=2, **kwargs):
    x, y, z = model.grid.dimensions
    t = model.grid.stepping_dim
    time = model.grid.time_dim
    s = time.spacing
    DTYPE = model.dtype
    savenum = rec.data.shape[0]

    v = VectorTimeFunction(name='v', grid=model.grid, space_order=space_order,
                           time_order=time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
    tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=space_order, 
                                time_order=time_order, dtype=DTYPE,
                                staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
    vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=space_order,
                               time_order=time_order, dtype=DTYPE, save=savenum)
    grad_k_func = Function(name='grad_k', grid=model.grid, space_order=space_order, dtype=DTYPE)
    grad_m_func = Function(name='grad_m', grid=model.grid, space_order=space_order, dtype=DTYPE)
    
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
    return Operator([u_vx, u_vy, u_vz]
                    +[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy,u_tzz]
                    +src_xx+src_yy+src_zz+[grad_ksum]+[grad_msum],
                    compiler='nvc',platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))

def JRAdjointcheckOperator(model, rec, src, space_order=10, time_order=2, checkpoint=False, **kwargs):
    x, y, z = model.grid.dimensions
    t = model.grid.stepping_dim
    time = model.grid.time_dim
    s = time.spacing
    DTYPE = model.dtype
    savenum = rec.data.shape[0]

    v_adj = VectorTimeFunction(name='v_adj', grid=model.grid, space_order=space_order,
                           time_order=time_order, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
    tau_adj = TensorTimeFunction(name='sigma_adj', grid=model.grid, space_order=space_order,
                             time_order=time_order, dtype=DTYPE,
                             staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])
    if checkpoint:
        vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=space_order,
                                   time_order=time_order, dtype=DTYPE)
    else:
        vsave = VectorTimeFunction(name='vsave', grid=model.grid, space_order=space_order,
                                   time_order=time_order, dtype=DTYPE, save=savenum)
    grad_k_func = Function(name='grad_k', grid=model.grid, space_order=space_order, dtype=DTYPE)
    grad_m_func = Function(name='grad_m', grid=model.grid, space_order=space_order, dtype=DTYPE)
    
    src_xx = src.inject(field=tau_adj.backward[0, 0], expr=-src)
    src_yy = src.inject(field=tau_adj.backward[1, 1], expr=-src)
    src_zz = src.inject(field=tau_adj.backward[2, 2], expr=-src)
    
    # Split coordinate pml
    u_vx = Eq(v_adj.backward[0], v_adj[0] * (1 - s*model.s_x - s*model.alp_x) + s*model.b_x*div(tau_adj)[0])
    u_vy = Eq(v_adj.backward[1], v_adj[1] * (1 - s*model.s_y - s*model.alp_y) + s*model.b_y*div(tau_adj)[1])
    u_vz = Eq(v_adj.backward[2], v_adj[2] * (1 - s*model.s_z - s*model.alp_z) + s*model.b_z*div(tau_adj)[2])

    u_txx = Eq(tau_adj.backward[0, 0], (tau_adj[0, 0] * (1 - s*model.s_x) 
                                    + s* (model.lam_diag*diag(div(v_adj.backward)) 
                                            + model.mu_diag * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[0, 0]))
    u_tyy = Eq(tau_adj.backward[1, 1], (tau_adj[1, 1] * (1 - s*model.s_y)
                                    + s* (model.lam_diag*diag(div(v_adj.backward)) 
                                            + model.mu_diag * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[1, 1]))
    u_tzz = Eq(tau_adj.backward[2, 2], (tau_adj[2, 2] * (1 - s*model.s_z) 
                                    + s* (model.lam_diag*diag(div(v_adj.backward)) 
                                            + model.mu_diag * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[2, 2]))

    u_tyx = Eq(tau_adj.backward[1, 0], (tau_adj[1, 0]*(1-s*model.s_y) +
                                    s*(model.mu_xy * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[1, 0]))
    u_txy = Eq(tau_adj.backward[0, 1], (tau_adj[0, 1]*(1-s*model.s_x) +
                                    s*(model.mu_xy * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[0, 1]))
    u_tzx = Eq(tau_adj.backward[2, 0], (tau_adj[2, 0]*(1-s*model.s_z) +
                                    s*(model.mu_xz * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[2, 0]))
    u_txz = Eq(tau_adj.backward[0, 2], (tau_adj[0, 2]*(1-s*model.s_x) +
                                    s*(model.mu_xz * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[0, 2]))
    u_tyz = Eq(tau_adj.backward[1, 2], (tau_adj[1, 2]*(1-s*model.s_y) +
                                    s*(model.mu_yz * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[1, 2]))
    u_tzy = Eq(tau_adj.backward[2, 1], (tau_adj[2, 1]*(1-s*model.s_z) +
                                    s*(model.mu_yz * (grad(v_adj.backward)+grad(v_adj.backward).transpose(inner=False)))[2, 1]))
    
    grad_ksum = Eq(grad_k_func, (grad_k_func + (grad(vsave)[0,0]+grad(vsave)[1,1]+grad(vsave)[2,2]) 
                                    * (tau_adj[0, 0]+tau_adj[1, 1]+tau_adj[2, 2])))
    grad_msum = Eq(grad_m_func, (grad_m_func + (4*tau_adj[0,0]/3-2*tau_adj[1,1]/3-2*tau_adj[2,2]/3)*grad(vsave)[0,0]
                                    + (4*tau_adj[1,1]/3-2*tau_adj[0,0]/3-2*tau_adj[2,2]/3)*grad(vsave)[1,1]
                                    + (4*tau_adj[2,2]/3-2*tau_adj[0,0]/3-2*tau_adj[1,1]/3)*grad(vsave)[2,2]
                                    + (tau_adj[0,1]+tau_adj[1,0])*(grad(vsave)[0,1]+grad(vsave)[1,0])
                                    + (tau_adj[1,2]+tau_adj[2,1])*(grad(vsave)[1,2]+grad(vsave)[2,1])
                                    + (tau_adj[2,0]+tau_adj[0,2])*(grad(vsave)[2,0]+grad(vsave)[0,2])))
    return Operator([u_vx, u_vy, u_vz]
                    +[u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy,u_tzz]
                    +src_xx+src_yy+src_zz+[grad_ksum]+[grad_msum],
                    compiler='nvc',platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))
