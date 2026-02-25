## Testing Devito single GPU case with homogeneous acoustic medium and time varying source
import numpy as np
import sys
import os
import argparse
import time
from scipy.io import savemat
from devito import (VectorTimeFunction, TensorTimeFunction, Function, Eq,
                    div, Operator, grad, diag)
sys.path.append('/home/hkhuang3/Documents/devito/examples/anastasio/')
from tranPACT import TranPACTModel, TimeAxis, TranPACTJRWaveSolver, Receiver, RickerSource
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
from devito import configuration
configuration['log-level'] = 'ERROR' #'DEBUG','INFO','ERROR'

parser = argparse.ArgumentParser('Proper usage.')
parser.add_argument('-s','--scale', type=float, default=5)
parser.add_argument('-t','--Nt', type=int, default=101)
parser.add_argument('-g','--gpu', type=int, default=1)
args = parser.parse_args()

if args.gpu==1:
    configuration['language'] = 'openacc'
    configuration['platform'] = 'nvidiaX'
    configuration['compiler'] = 'nvc'
    configuration['ignore-unknowns'] = 1

## parameter setting
DTYPE = np.float32
fn = np.array([[1000.0,2190.4,0.0,0.0],[1000.0,2190.4,0.0,0.0]])
# Wavelength, frequency, and Sampling rate
cb = 1.5
f0 = 1.0e0 # MHz
ll = cb / f0 # mm
ppw = 3
# Volume dimensions
based_size_unit = args.scale

xlen = based_size_unit * ll # Size of simulation in mm
ylen = based_size_unit * ll # Size of simulation in mm
zlen = based_size_unit * ll # Size of simulation in mm
# Grid dimensions
dx, dy, dz = ll / ppw, ll / ppw, ll / ppw
fs = 10
dt = 1/fs

# Simulation order
so = 10
to = 2
# ABC thickness
nbl = 16
# parameters for grid defining
Nx = int(np.round(xlen / dx))
Ny = int(np.round(ylen / dy))
Nz = int(np.round(zlen / dz))
shape = (Nx, Ny, Nz)
origin = (0, 0, 0)
spacing = (dx, dy, dz)

## for medium geometry setting(cylinder shell)
r_dis = np.zeros((Nx,Ny,Nz),dtype=np.float32)

## for specifying optimization ROI
roimode = 1
opt_roi = np.ones(shape,dtype=np.bool_)

med_rho_static = np.zeros_like(r_dis)
## Creating model
model = TranPACTModel(space_order=so, medium_geometry=r_dis, medium_param=fn,
                        inhouse_med=False,origin=origin, opt_roi=opt_roi, water_index=0,
                        shape=shape, spacing=spacing, nbl=nbl, dtype=DTYPE, med_rho=med_rho_static)

## Setting time and model
Nt = args.Nt
time_range = TimeAxis(start=0, num=Nt, step=dt)

rec_pos = np.zeros((1,3),dtype='f')
rec_pos[0, 0] = xlen / 2 + 1*ll
rec_pos[0, 1] = ylen / 2
rec_pos[0, 2] = zlen / 2
rec = Receiver(name="rec", grid=model.grid, npoint=1, time_range=time_range, dtype=DTYPE)
rec.coordinates.data[:,:] = rec_pos

src = RickerSource(name='src', grid=model.grid, f0=f0/3, npoint=1, time_range=time_range, dtype=DTYPE)
src_pos = np.zeros((1,3),dtype='f')
src_pos[0, 0] = xlen / 2 - 1*ll
src_pos[0, 1] = ylen / 2
src_pos[0, 2] = zlen / 2
src.coordinates.data[:,:] = src_pos

solver = TranPACTJRWaveSolver(model, rec, src, time_range, np.zeros((Nx,Ny,Nz),dtype=np.float32), to)

x, y, z = model.grid.dimensions
t = model.grid.stepping_dim
time = model.grid.time_dim
s = time.spacing
DTYPE = model.dtype
nbl = model.nbl
opt_roi = model.opt_roi
(Nx,Ny,Nz) = model.shape

v = VectorTimeFunction(name='v', grid=model.grid, space_order=so,
                       time_order=to, dtype=DTYPE, staggered=[(-x, -t), (y, -t), (z, -t)])
tau = TensorTimeFunction(name='sigma', grid=model.grid, space_order=so, 
                         time_order=to, dtype=DTYPE,
                         staggered=[[(), (-x,y), (-x,z)], [(-x,y), (), (y,z)], [(-x,z), (y,z), ()]])

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

src_xx = src.inject(field=tau.forward[0, 0], expr=-src * s * cb / (dx * dy * dz))
src_yy = src.inject(field=tau.forward[1, 1], expr=-src * s * cb / (dx * dy * dz))
src_zz = src.inject(field=tau.forward[2, 2], expr=-src * s * cb / (dx * dy * dz))

if args.gpu==1:
    op = Operator([u_vx, u_vy, u_vz] + [u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz]+ rec_term+src_xx+src_yy+src_zz, compiler='nvc', platform='nvidiaX', language='openacc', opt=('advanced', {'openmp' : True}))
else:
    op = Operator([u_vx, u_vy, u_vz] + [u_txx, u_txy, u_txz, u_tyx, u_tyy, u_tyz, u_tzx, u_tzy, u_tzz] + rec_term+src_xx+src_yy+src_zz, opt=('advanced', {'openmp' : True}))
op.apply(dt=dt, time_m=0, time_M=time_range.num-1)

p0_forward = rec.data[:,:]

mdic = {"p0_forward": p0_forward}
# savemat(f"forward.mat", mdic)
