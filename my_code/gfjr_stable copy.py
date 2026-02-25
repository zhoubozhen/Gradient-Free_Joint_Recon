#!/usr/bin/env python3
import os
import sys
import argparse
import json
import time
import threading
import subprocess
from datetime import datetime
import math

import numpy as np
import scipy.io as sio
import h5py

# print(">>> BEFORE DEBUG BREAK")
# breakpoint()

# ==========================================================
# 0) 强制只使用 V2_implement/src 里的新版代码（最先）
# ==========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# ==========================================================
# 1) parse args（必须在 import devito / cupy 之前）
# ==========================================================
parser = argparse.ArgumentParser('Recon data')
parser.add_argument('-g', '--gpu', type=str, default=None)
parser.add_argument('--mpi', action='store_true')
parser.add_argument('-i', '--ind', type=int, default=3)
parser.add_argument('--skullp0', type=int, default=1)
parser.add_argument('--onlycor', action='store_true')
parser.add_argument('--pressure', type=str, required=True)
parser.add_argument('-n', '--noise', type=int, default=5)
parser.add_argument('-r', '--reg', type=float, default=0.01)
parser.add_argument('-l', '--lip', type=float, default=5.0)
parser.add_argument('--recon_opt', type=int, required=True)
parser.add_argument('--iter', type=int, default=50)
parser.add_argument('--maxfun', type=int, default=10,
                    help='max function evaluations for pybobyqa')
parser.add_argument('--stride', type=float, default=4/3)
parser.add_argument('--start', type=float, default=1)
parser.add_argument('--prox_mode', type=int, default=1,
                    help='0:CPU proximal_L, 1:cupy_mix, 2:cupy, else:2d')

args = parser.parse_args()

# ==========================================================
# 2) DEBUG/TRACE 开关（full scale 定位卡点用）
# ==========================================================
DEBUG_SMALL = True             # <- 你要 full scale 就保持 False
DEBUG_ONLY_ONE_FISTA = False     # True: 只跑一次 inner FISTA 然后退出
# DEBUG_STRIDE = 1                 # 允许非整数（例如 ppw=3 时可设 4/3）
DEBUG_STRIDE = args.stride

if DEBUG_STRIDE==1:
    DEBUG_SMALL = False

DEBUG_NT = 4800

TRACE_FULL = True               # <- 强制开详细日志（full scale 必开）
HEARTBEAT_SEC = 60              # 心跳间隔（秒）
LOG_GPU = True                  # 是否查询 nvidia-smi

# ==========================================================
# 3) GPU / MPI 绑定（早于 devito）
# ==========================================================
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

comm = None
rank = 0
size = 1

if args.mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
else:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==========================================================
# 4) import 项目依赖
# ==========================================================
from tranPACT import TranPACTModel, TranPACTWaveSolver, GFJRSolver, OptimParam
from devito import configuration

configuration['language'] = 'openacc'
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'nvc'
configuration['log-level'] = 'INFO'
configuration['ignore-unknowns'] = 1
if args.mpi:
    configuration['mpi'] = True

# ==========================================================
# 5) 一些工具函数：时间戳 / 内存 / GPU / 日志
# ==========================================================
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _read_proc_status():
    out = {}
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:", "VmSize:")):
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
    except Exception:
        pass
    return out

def _gpu_snapshot():
    if not LOG_GPU:
        return ""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        s = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        gpu_strs = []
        for line in s.splitlines():
            gid, util, mem_used, mem_total = line.split(",")
            mem_used = int(mem_used) / 1024
            mem_total = int(mem_total) / 1024
            gpu_strs.append(
                f"{gid.strip()}: util={util.strip()}% mem={mem_used:.1f}/{mem_total:.1f}GB"
            )
        return " | GPU: " + " ; ".join(gpu_strs)
    except Exception:
        return ""

LOG_FH = None

def _kb_to_gb(s):
    try:
        return f"{int(s.split()[0]) / (1024**2):.2f} GB"
    except Exception:
        return s

def log(msg):
    global LOG_FH
    prefix = f"[{_now()}][rank {rank}/{size}][pid {os.getpid()}]"
    st = _read_proc_status()
    mem = ""
    if st:
        mem = (
            f" | RSS={_kb_to_gb(st.get('VmRSS','?'))}"
            f" Swap={_kb_to_gb(st.get('VmSwap','?'))}"
            f" HWM={_kb_to_gb(st.get('VmHWM','?'))}"
        )
    g = _gpu_snapshot()
    line = f"{prefix} {msg}{mem}{g}"
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()

class StageTimer:
    def __init__(self, name):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        log(f"--> ENTER {self.name}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc is None:
            log(f"<-- EXIT  {self.name}  dt={dt:.3f}s")
        else:
            log(f"<!! EXC   {self.name}  dt={dt:.3f}s  exc={exc_type.__name__}: {exc}")
        return False

# 心跳线程：即使卡住，也能看到“卡住时 RSS/GPU 的状态”
_stop_hb = False
def _heartbeat_loop():
    while not _stop_hb:
        log("HEARTBEAT (alive)")
        time.sleep(HEARTBEAT_SEC)

# ==========================================================
# 5.1) DEBUG_SMALL: 物理正确下采样索引（支持非整数 stride）
# ==========================================================
def downsample_index(n: int, stride: float) -> np.ndarray:
    """
    stride 可为 float。
    返回长度约为 ceil(n/stride) 的索引数组（nearest），保证覆盖整个物理范围。
    不做插值，不改变物理长度；物理长度由 spacing * n_new 来保持。
    """
    stride = float(stride)
    if stride <= 1.0:
        return np.arange(n, dtype=np.int64)
    n_new = int(math.ceil(n / stride))
    idx = np.round(np.linspace(0, n - 1, n_new)).astype(np.int64)
    # 去重并保持递增
    idx = np.unique(idx)
    return idx

def apply_downsample_3d(arr: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    return arr[np.ix_(ix, iy, iz)]

# ==========================================================
# 6) 先读 rec_pos → 确定 Nrec（关键）
# ==========================================================
DTYPE = np.float32
noise_level = args.noise
reg_param = args.reg
lip = args.lip
recon_opt = args.recon_opt

nhp_directory = (
    '/shared/anastasio-s3/CommonData/bozhen/gfjr_data/'
    'kevin_rsaas/KH250319_headphantom/data/'
)

directory = (
    '/shared/anastasio-s3/CommonData/bozhen/'
    'gfjr_data/kevin_rsaas/KH250727_JRwithAubry/'
)

with StageTimer("load rec_pos"):
    rec_pos = np.fromfile(nhp_directory + 'mp_raw.DAT', dtype=DTYPE).reshape(3, -1).T
    rec_pos -= np.min(rec_pos, axis=0) - 1.5
Nrec = rec_pos.shape[0] # 51200
log(f"rec_pos loaded: Nrec={Nrec}")

# ==========================================================
# 7) 读取 pressure（支持 1D flatten）+ truncate Nt
# ==========================================================
with StageTimer("load forward pressure"):
    with h5py.File(directory + f'vit_data/{args.pressure}.hdf5', 'r') as f:
        p0_forward = f['forward'][:].astype(np.float32)

if p0_forward.ndim == 1:
    Nt_full = p0_forward.size // Nrec
    assert Nt_full * Nrec == p0_forward.size
    p0_forward = p0_forward.reshape(Nt_full, Nrec)
elif p0_forward.ndim == 2:
    Nt_full, _ = p0_forward.shape
else:
    raise ValueError(f"Unexpected forward ndim={p0_forward.ndim}")

Nt = DEBUG_NT if DEBUG_SMALL else Nt_full
p0_forward = np.ascontiguousarray(p0_forward[:Nt, :], dtype=np.float32)

log(f"Loaded forward: Nt_full={Nt_full}, Nrec={Nrec}, use Nt={Nt}, bytes={p0_forward.nbytes/1e9:.3f}GB")

# ==========================================================
# 8) medium 参数
# ==========================================================
with StageTimer("load fn"):
    if recon_opt == 2:
        fn = np.fromfile(directory + 'param/fn_water_bone.DAT', dtype=DTYPE).reshape(-1, 4)
    elif recon_opt == 1:
        fn = np.fromfile(directory + 'param/fn_1p3l_nhp_3.DAT', dtype=DTYPE).reshape(-1, 4)
    else:
        fn = np.fromfile(directory + 'param/fn_1p1l_nhp_3.DAT', dtype=DTYPE).reshape(-1, 4)
log(f"fn shape={fn.shape}")

# ==========================================================
# 9) 网格 & ROI（full / small） —— 仅在这里实现“物理正确下采样”
# ==========================================================
cb = 1.5
f0 = 1.0
ppw = 4
ll = cb / f0
dx0 = dy0 = dz0 = ll / ppw

fs = 30
dt = 1 / fs
so = 10
to = 2
nbl = 16

with StageTimer("load opt_roi"):
    opt_roi = np.fromfile(nhp_directory + 'param/roi_intran.DAT', dtype=DTYPE).reshape(702, 702, 350)
    cor_roi = np.load(nhp_directory + 'param/roi_cor.npy')
    skull_roi = np.load(nhp_directory + 'param/roi_diluted_skull.npy')

if args.skullp0 == 0:
    opt_roi[skull_roi == 1] = 0
if args.onlycor:
    opt_roi[cor_roi == 0] = 0

# DEBUG_SMALL：对 ROI 做 nearest 取样（支持非整数 stride），并同步放大 spacing
if DEBUG_SMALL:
    s = float(DEBUG_STRIDE)
    ix = downsample_index(opt_roi.shape[0], s)
    iy = downsample_index(opt_roi.shape[1], s)
    iz = downsample_index(opt_roi.shape[2], s)
    opt_roi = apply_downsample_3d(opt_roi, ix, iy, iz)

    dx = dx0 * s
    dy = dy0 * s
    dz = dz0 * s
else:
    ix = iy = iz = None
    dx, dy, dz = dx0, dy0, dz0

opt_roi = np.ascontiguousarray(opt_roi.astype(np.float32))
Nx, Ny, Nz = opt_roi.shape
shape = (Nx, Ny, Nz)
log(f"ROI shape={shape}, nnz={int(np.count_nonzero(opt_roi))}, spacing=({dx:.6f},{dy:.6f},{dz:.6f}), bytes={opt_roi.nbytes/1e9:.3f}GB")

# ==========================================================
# 10) model + solver
# ==========================================================
with StageTimer("build model"):
    if recon_opt == 0:
        acnhp = sio.loadmat(nhp_directory + 'acoustic/nhp_1p1l_vifov.mat')['annhp']
        if DEBUG_SMALL:
            acnhp = apply_downsample_3d(acnhp, ix, iy, iz)
        model = TranPACTModel(
            space_order=so, medium_geometry=acnhp,
            medium_param=fn, water_index=0,
            medium_mode='mpml', origin=(0, 0, 0),
            opt_roi=opt_roi, shape=shape,
            spacing=(dx, dy, dz), nbl=nbl, dtype=DTYPE
        )
    elif recon_opt == 1:
        acnhp = sio.loadmat(nhp_directory + 'acoustic/nhp_1p3l_vifov.mat')['annhp']
        if DEBUG_SMALL:
            acnhp = apply_downsample_3d(acnhp, ix, iy, iz)
        model = TranPACTModel(
            space_order=so, medium_geometry=acnhp,
            medium_param=fn, water_index=0,
            medium_mode='mpml', origin=(0, 0, 0),
            opt_roi=opt_roi, shape=shape,
            spacing=(dx, dy, dz), nbl=nbl, dtype=DTYPE
        )
    else:
        mdic = sio.loadmat(directory + 'param/nhp_ctdata_vifov.mat')
        ct_data = mdic['ct_data']
        use_static = True
        if DEBUG_STRIDE == 1:
            use_downsample = False
        else:
            use_downsample = True
        model = TranPACTModel(
            space_order=so, medium_param=fn,
            water_index=0, ct_data=ct_data,
            cor_length=1, medium_mode='aubry',
            origin=(0, 0, 0), opt_roi=opt_roi,
            shape=shape, spacing=(dx, dy, dz),
            nbl=nbl, dtype=DTYPE, use_static=use_static, use_downsample=use_downsample
        )

with StageTimer("build solver"):
    solver = TranPACTWaveSolver(model, rec_pos=rec_pos, Nt=Nt, dt=dt, to=to)

# ==========================================================
# 11) 噪声（同样 reshape + truncate）
# ==========================================================
with StageTimer("load noise"):
    with h5py.File(nhp_directory + 'param/noise_filtered_unit_std.hdf5', 'r') as f:
        p0_noise = f['forward'][:].astype(np.float32)

if p0_noise.ndim == 1:
    Nt_noise_full = p0_noise.size // Nrec
    p0_noise = p0_noise.reshape(Nt_noise_full, Nrec)
p0_noise = np.ascontiguousarray(p0_noise[:Nt, :], dtype=np.float32)

if noise_level != 0:
    with StageTimer("add noise"):
        p0_forward += p0_noise * (0.00026 * noise_level)

# ==========================================================
# 12) saving_dir + 初始化 LOG_FH
# ==========================================================
saving_root = "/shared/anastasio-s3/CommonData/bozhen/gfjr_data/case_study"
# tag = datetime.now().strftime("%Y%m%d")
saving_dir = os.path.join(saving_root, f"{args.pressure}_fulltrace")

if rank == 0:
    os.makedirs(saving_dir, exist_ok=True)
    with open(os.path.join(saving_dir, 'args_record.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

os.makedirs(saving_dir, exist_ok=True)
LOG_FH = open(os.path.join(saving_dir, f"run_fulltrace_rank{rank}.log"), "w", buffering=1)
log(f"LOG_FILE = {os.path.join(saving_dir, f'run_fulltrace_rank{rank}.log')}")

# ==========================================================
# 13) 包一层 forward/adjoint wrapper（只能定位到 devito 调用边界）
# ==========================================================
def forward_wrapped(p0):
    with StageTimer("DEVITO forward()"):
        out = solver.forward(p0)
    log(f"forward out: shape={getattr(out,'shape',None)} dtype={getattr(out,'dtype',None)}")
    return out

def adjoint_wrapped(p):
    with StageTimer("DEVITO adjoint()"):
        out = solver.adjoint(p)
    log(f"adjoint out: shape={getattr(out,'shape',None)} dtype={getattr(out,'dtype',None)}")
    return out

# ==========================================================
# 14) GFJR / FISTA 参数
# ==========================================================
opt_param = OptimParam(
    reg=reg_param,
    num_iter=args.iter,
    out_print=3,
    lip=lip,
    saving_dir=saving_dir,
    use_check=False,
    save_freq=1 
)
opt_param.prox_mode=args.prox_mode
if DEBUG_STRIDE == 1:
    use_downsample = False
else:
    use_downsample = True
gfjrsolver = GFJRSolver(
    solver=solver,
    measured_pressure=p0_forward,
    fn=fn,
    forward=forward_wrapped,
    adjoint=adjoint_wrapped,
    opt_param=opt_param,
    comm=comm,
    Nt=Nt,use_downsample=use_downsample
)

# 心跳线程：full scale 时必须开
if TRACE_FULL:
    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()

# ==========================================================
# 15) 只跑一次 inner FISTA（最快定位卡在 FISTA 哪一步）
# ==========================================================
if DEBUG_ONLY_ONE_FISTA:
    log("DEBUG_ONLY_ONE_FISTA=True: run one inner FISTA then exit")
    with StageTimer("RUN inner FISTA"):
        p0_test, cost, it = gfjrsolver.cost_function.inner_solver(init_guess=gfjrsolver.p0_est)
    log(f"[DEBUG] inner FISTA finished: cost={cost} it={it} p0.shape={p0_test.shape}")
    _stop_hb = True
    sys.exit(0)

# ==========================================================
# 16) GFJR solve
# ==========================================================
c0 = np.array([4.0, 2.0], dtype=DTYPE) * args.start # 0.98
lower = 0.9 * c0
upper = 1.1 * c0

log("START GFJR.solve()")
with StageTimer("GFJR.solve"):
    gfjrsolver.solve(c0, lower, upper, num_iter_init=args.iter * 2, maxfun=args.maxfun)

_stop_hb = True
log("ALL DONE")
