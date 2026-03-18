#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import scipy.io as sio
import h5py

# -------------------------
# config helpers
# -------------------------
def _deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_repo_root_from_file() -> str:
    # .../repo_root/new_v2/my_code/main.py -> repo_root
    this_file = os.path.abspath(__file__)
    my_code_dir = os.path.dirname(this_file)
    new_v2_dir = os.path.dirname(my_code_dir)
    repo_root = os.path.dirname(new_v2_dir)
    return repo_root

def _set_env_before_cupy(cfg: dict):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # MPI: local rank -> CUDA_VISIBLE_DEVICES=local_rank（只用“可见卡”的局部编号）
    if cfg.get("mpi", False):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        return comm, rank, size

    # non-MPI:
    # 1) 若外部已显式设置 CUDA_VISIBLE_DEVICES（如 cluster/run.sh），则尊重外部设置
    # 2) 否则优先使用 binding.main_gpu_idx
    # 3) 最后兼容旧字段 gpu
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        binding_cfg = dict(cfg.get("binding", {}) or {})
        main_gpu_idx = binding_cfg.get("main_gpu_idx", None)
        if main_gpu_idx is not None and str(main_gpu_idx) != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(main_gpu_idx)
            os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(main_gpu_idx))
        else:
            gpu = cfg.get("gpu", None)
            if gpu is not None and str(gpu) != "":
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(gpu))

    return None, 0, 1

def _set_openacc_env():
    # main 进程只暴露 1 张卡时，device_num=0 就是那张
    os.environ.setdefault("NV_ACC_DEVICE_TYPE", "NVIDIA")
    os.environ.setdefault("NV_ACC_DEVICE_NUM", "0")
    os.environ.setdefault("ACC_DEVICE_NUM", "0")

def _print_cupy_device():
    import cupy as cp
    print("[MAIN] cupy device count    =", cp.cuda.runtime.getDeviceCount(), flush=True)
    print("[MAIN] cupy current device  =", cp.cuda.Device().id, flush=True)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else str(props["name"])
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[MAIN] cupy device name     = {name}", flush=True)
    print(f"[MAIN] cupy mem free/total  = {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)

# -------------------------
# entry
# -------------------------
def main():
    ap = argparse.ArgumentParser("GFJR main (config-driven)")
    ap.add_argument("--config", required=True, help="Path to config JSON")
    # 可选：CLI 覆盖（只给最常用的几个；需要更多再加）
    ap.add_argument("--pressure", type=str, default=None)
    ap.add_argument("--recon_opt", type=int, default=None)
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--maxfun", type=int, default=None)
    ap.add_argument("--lip", type=float, default=None)
    ap.add_argument("--reg", type=float, default=None)
    ap.add_argument("--stride", type=float, default=None)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--prox_mode", type=int, default=None)
    ap.add_argument("--prox_impl", type=str, default=None)
    ap.add_argument("--mpi", action="store_true")
    args = ap.parse_args()

    cfg = _load_json(args.config)

    # CLI 覆盖 config（最少维护成本）
    cli_override = {}
    for k in ["pressure", "recon_opt", "iter", "maxfun", "lip", "reg", "stride", "start", "prox_mode", "prox_impl"]:
        v = getattr(args, k)
        if v is not None:
            cli_override[k] = v
    if args.mpi:
        cli_override["mpi"] = True
    cfg = _deep_update(cfg, cli_override)

    # 必填检查
    for need in ["pressure", "recon_opt"]:
        if need not in cfg or cfg[need] in [None, ""]:
            raise SystemExit(f"ERROR: config missing required field: {need}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    src_dir = os.path.join(package_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 工具（你已有 gfjr_utils.py）
    from gfjr_utils import (
        print_main_env_snapshot,
        try_print_nvidia_smi,
        Logger,
        downsample_index,
        apply_downsample_3d,
    )

    # 先设置 env（必须早于 cupy/devito）
    comm, rank, size = _set_env_before_cupy(cfg)
    _set_openacc_env()

    print_main_env_snapshot("MAIN")
    try_print_nvidia_smi("MAIN")

    # import cupy（必须在 env 之后）
    _print_cupy_device()

    # devito config
    from devito import configuration
    configuration["language"] = "openacc"
    configuration["platform"] = "nvidiaX"
    configuration["compiler"] = "nvc"
    configuration["log-level"] = cfg.get("devito_log_level", "INFO")
    configuration["ignore-unknowns"] = 1
    if cfg.get("mpi", False):
        configuration["mpi"] = True

    # runtime flags（来自 config）
    runtime = cfg.get("runtime", {})
    DEBUG_SMALL = bool(runtime.get("debug_small", True))
    DEBUG_ONLY_ONE_FISTA = bool(runtime.get("debug_only_one_fista", False))
    DEBUG_NT = int(runtime.get("debug_nt", 4800))
    TRACE_FULL = bool(runtime.get("trace_full", True))
    HEARTBEAT_SEC = int(runtime.get("heartbeat_sec", 60))
    LOG_GPU = bool(runtime.get("log_gpu", True))

    stride = float(cfg.get("stride", 4/3))
    if stride == 1.0:
        DEBUG_SMALL = False

    # 常用实验参数
    pressure = cfg["pressure"]
    recon_opt = int(cfg["recon_opt"])
    ind = int(cfg.get("ind", 3))
    skullp0 = int(cfg.get("skullp0", 1))
    onlycor = bool(cfg.get("onlycor", False))
    noise_level = int(cfg.get("noise", 5))
    maxfun = int(cfg.get("maxfun", 10))
    start = float(cfg.get("start", 1.0))
    fista_cfg = cfg.get("fista", {})
    prox_mode = int(fista_cfg.get("prox_mode", 1))
    prox_impl = str(fista_cfg.get("prox_impl", "mix")).strip().lower()
    reg_param = float(fista_cfg.get("reg", 0.0001))
    lip = float(fista_cfg.get("lip", 5.0))
    num_iter = int(fista_cfg.get("iter", 20))
    save_freq = int(fista_cfg.get("save_freq", 1))
    use_check = bool(fista_cfg.get("use_check", False))
    grad_min = float(fista_cfg.get("grad_min", 1e-5))
    grad_min_init = float(fista_cfg.get("grad_min_init", 1e-4))
    cost_min = float(fista_cfg.get("cost_min", 1e-3))
    check_iter = int(fista_cfg.get("check_iter", 10))
    prox_iter = int(fista_cfg.get("prox_iter", 50))
    rel_thr = float(fista_cfg.get("rel_thr", 1e-3))
    rel_patience = int(fista_cfg.get("rel_patience", 2))
    rel_warmup = int(fista_cfg.get("rel_warmup", 2))
    div_rel_thr = float(fista_cfg.get("div_rel_thr", 1e-2))
    div_patience = int(fista_cfg.get("div_patience", 2))
    div_warmup = int(fista_cfg.get("div_warmup", 2))

    fista_runtime_cfg = dict(fista_cfg.get("runtime", {}) or {})
    worker_script = fista_runtime_cfg.get("worker_script", None)

    binding_cfg = dict(cfg.get("binding", {}) or {})
    prox_gpu_idx = binding_cfg.get("prox_gpu_idx", None)

    # cluster: 优先读 .sh / Condor 导出的环境变量
    prox_cuda_visible_devices = os.environ.get("PROX_CUDA_VISIBLE_DEVICES", "").strip() or None
    prox_nvidia_visible_devices = os.environ.get("PROX_NVIDIA_VISIBLE_DEVICES", "").strip() or None

    # local: 如果环境变量没给，再回退到 config 里的 prox_gpu_idx
    if prox_cuda_visible_devices is None and prox_nvidia_visible_devices is None:
        if prox_gpu_idx is not None and str(prox_gpu_idx).strip() != "":
            prox_cuda_visible_devices = str(prox_gpu_idx)
            prox_nvidia_visible_devices = str(prox_gpu_idx)

    # paths（来自 config）
    paths = cfg.get("paths", {})
    nhp_directory = paths["nhp_directory"]
    directory = paths["directory"]
    saving_root = paths["saving_root"]

    # logger
    logger = Logger(rank=rank, size=size, log_gpu=LOG_GPU, heartbeat_sec=HEARTBEAT_SEC)
    log = logger.log
    StageTimer = logger.stage

    # import project deps
    from tranPACT import TranPACTModel, TranPACTWaveSolver, GFJRSolver, OptimParam

    # -------------------------
    # 7) load rec_pos
    # -------------------------
    DTYPE = np.float32
    with StageTimer("load rec_pos"):
        rec_pos = np.fromfile(os.path.join(nhp_directory, "mp_raw.DAT"), dtype=DTYPE).reshape(3, -1).T
        rec_pos -= np.min(rec_pos, axis=0) - 1.5
    Nrec = rec_pos.shape[0]
    log(f"rec_pos loaded: Nrec={Nrec}")

    # -------------------------
    # 8) load pressure
    # -------------------------
    with StageTimer("load forward pressure"):
        with h5py.File(os.path.join(directory, f"vit_data/{pressure}.hdf5"), "r") as f:
            p0_forward = f["forward"][:].astype(np.float32)

    if p0_forward.ndim == 1:
        Nt_full = p0_forward.size // Nrec
        if Nt_full * Nrec != p0_forward.size:
            raise RuntimeError("forward data size mismatch with Nrec")
        p0_forward = p0_forward.reshape(Nt_full, Nrec)
    elif p0_forward.ndim == 2:
        Nt_full, _ = p0_forward.shape
    else:
        raise ValueError(f"Unexpected forward ndim={p0_forward.ndim}")

    Nt = DEBUG_NT if DEBUG_SMALL else Nt_full
    p0_forward = np.ascontiguousarray(p0_forward[:Nt, :], dtype=np.float32)
    log(f"Loaded forward: Nt_full={Nt_full}, Nrec={Nrec}, use Nt={Nt}, bytes={p0_forward.nbytes/1e9:.3f}GB")

    # -------------------------
    # 9) fn
    # -------------------------
    with StageTimer("load fn"):
        if recon_opt == 2:
            fn = np.fromfile(os.path.join(directory, "param/fn_water_bone.DAT"), dtype=DTYPE).reshape(-1, 4)
        elif recon_opt == 1:
            fn = np.fromfile(os.path.join(directory, "param/fn_1p3l_nhp_3.DAT"), dtype=DTYPE).reshape(-1, 4)
        else:
            fn = np.fromfile(os.path.join(directory, "param/fn_1p1l_nhp_3.DAT"), dtype=DTYPE).reshape(-1, 4)
    log(f"fn shape={fn.shape}")

    # -------------------------
    # 10) grid & ROI
    # -------------------------
    cb = float(cfg.get("physics", {}).get("cb", 1.5))
    f0 = float(cfg.get("physics", {}).get("f0", 1.0))
    ppw = int(cfg.get("physics", {}).get("ppw", 4))
    ll = cb / f0
    dx0 = dy0 = dz0 = ll / ppw

    fs = float(cfg.get("physics", {}).get("fs", 30))
    dt = 1.0 / fs
    so = int(cfg.get("physics", {}).get("space_order", 10))
    to = float(cfg.get("physics", {}).get("to", 2))
    nbl = int(cfg.get("physics", {}).get("nbl", 16))

    with StageTimer("load opt_roi"):
        opt_roi = np.fromfile(os.path.join(nhp_directory, "param/roi_intran.DAT"), dtype=DTYPE).reshape(702, 702, 350)
        cor_roi = np.load(os.path.join(nhp_directory, "param/roi_cor.npy"))
        skull_roi = np.load(os.path.join(nhp_directory, "param/roi_diluted_skull.npy"))

    if skullp0 == 0:
        opt_roi[skull_roi == 1] = 0
    if onlycor:
        opt_roi[cor_roi == 0] = 0

    if DEBUG_SMALL:
        s = float(stride)
        ix = downsample_index(opt_roi.shape[0], s)
        iy = downsample_index(opt_roi.shape[1], s)
        iz = downsample_index(opt_roi.shape[2], s)
        opt_roi = apply_downsample_3d(opt_roi, ix, iy, iz)
        dx, dy, dz = dx0 * s, dy0 * s, dz0 * s
    else:
        ix = iy = iz = None
        dx, dy, dz = dx0, dy0, dz0

    opt_roi = np.ascontiguousarray(opt_roi.astype(np.float32))
    Nx, Ny, Nz = opt_roi.shape
    shape = (Nx, Ny, Nz)
    log(f"ROI shape={shape}, nnz={int(np.count_nonzero(opt_roi))}, spacing=({dx:.6f},{dy:.6f},{dz:.6f}), bytes={opt_roi.nbytes/1e9:.3f}GB")

    # -------------------------
    # 11) model + solver
    # -------------------------
    with StageTimer("build model"):
        if recon_opt == 0:
            medium_geo_index = sio.loadmat(os.path.join(nhp_directory, "acoustic/nhp_1p1l_vifov.mat"))["annhp"]
            if DEBUG_SMALL:
                medium_geo_index = apply_downsample_3d(medium_geo_index, ix, iy, iz)
            model = TranPACTModel(
                space_order=so, medium_geometry=medium_geo_index,
                medium_param=fn, water_index=0,
                medium_mode="mpml", origin=(0, 0, 0),
                opt_roi=opt_roi, shape=shape,
                spacing=(dx, dy, dz), nbl=nbl, dtype=DTYPE
            )
        elif recon_opt == 1:
            medium_geo_index = sio.loadmat(os.path.join(nhp_directory, "acoustic/nhp_1p3l_vifov.mat"))["annhp"]
            if DEBUG_SMALL:
                medium_geo_index = apply_downsample_3d(medium_geo_index, ix, iy, iz)
            model = TranPACTModel(
                space_order=so, medium_geometry=medium_geo_index,
                medium_param=fn, water_index=0,
                medium_mode="mpml", origin=(0, 0, 0),
                opt_roi=opt_roi, shape=shape,
                spacing=(dx, dy, dz), nbl=nbl, dtype=DTYPE
            )
        else:
            mdic = sio.loadmat(os.path.join(directory, "param/nhp_ctdata_vifov.mat"))
            ct_data = mdic["ct_data"]
            use_static = True
            use_downsample = (stride != 1.0)
            model = TranPACTModel(
                space_order=so, medium_param=fn,
                water_index=0, ct_data=ct_data,
                cor_length=1, medium_mode="aubry",
                origin=(0, 0, 0), opt_roi=opt_roi,
                shape=shape, spacing=(dx, dy, dz),
                nbl=nbl, dtype=DTYPE, use_static=use_static, use_downsample=use_downsample
            )

    with StageTimer("build solver"):
        solver = TranPACTWaveSolver(model, rec_pos=rec_pos, Nt=Nt, dt=dt, to=to)

    # -------------------------
    # 12) noise
    # -------------------------
    with StageTimer("load noise"):
        with h5py.File(os.path.join(nhp_directory, "param/noise_filtered_unit_std.hdf5"), "r") as f:
            p0_noise = f["forward"][:].astype(np.float32)

    if p0_noise.ndim == 1:
        Nt_noise_full = p0_noise.size // Nrec
        p0_noise = p0_noise.reshape(Nt_noise_full, Nrec)
    p0_noise = np.ascontiguousarray(p0_noise[:Nt, :], dtype=np.float32)

    if noise_level != 0:
        with StageTimer("add noise"):
            noise_scale = float(cfg.get("noise_scale", 0.00026))
            p0_forward += p0_noise * (noise_scale * noise_level)

    # -------------------------
    # 13) saving dir + log file
    # -------------------------
    saving_dir = os.path.join(saving_root, f"{pressure}_fulltrace")
    os.makedirs(saving_dir, exist_ok=True)

    if rank == 0:
        with open(os.path.join(saving_dir, "args_record.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    log_path = os.path.join(saving_dir, f"run_fulltrace_rank{rank}.log")
    logger.set_log_file(log_path)
    log(f"LOG_FILE = {log_path}")

    # -------------------------
    # 14) forward/adjoint wrapper
    # -------------------------
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

    # -------------------------
    # 15) GFJR / FISTA params
    # -------------------------
    opt_param = OptimParam(
        reg=reg_param,
        num_iter=num_iter,
        out_print=int(cfg.get("out_print", 3)),
        lip=lip,
        saving_dir=saving_dir,
        use_check=use_check,
        save_freq=save_freq,
    )
    opt_param.prox_mode = prox_mode
    opt_param.prox_impl = prox_impl
    opt_param.grad_min = grad_min
    opt_param.grad_min_init = grad_min_init
    opt_param.cost_min = cost_min
    opt_param.check_iter = check_iter
    opt_param.prox_iter = prox_iter
    opt_param.rel_thr = rel_thr
    opt_param.rel_patience = rel_patience
    opt_param.rel_warmup = rel_warmup
    opt_param.div_rel_thr = div_rel_thr
    opt_param.div_patience = div_patience
    opt_param.div_warmup = div_warmup
    opt_param.worker_script = worker_script
    opt_param.prox_cuda_visible_devices = prox_cuda_visible_devices
    opt_param.prox_nvidia_visible_devices = prox_nvidia_visible_devices

    use_downsample = (stride != 1.0)
    gfjrsolver = GFJRSolver(
        solver=solver,
        measured_pressure=p0_forward,
        fn=fn,
        forward=forward_wrapped,
        adjoint=adjoint_wrapped,
        opt_param=opt_param,
        comm=comm,
        Nt=Nt,
        use_downsample=use_downsample,
    )

    if TRACE_FULL:
        logger.start_heartbeat()

    if DEBUG_ONLY_ONE_FISTA:
        log("DEBUG_ONLY_ONE_FISTA=True: run one inner FISTA then exit")
        with StageTimer("RUN inner FISTA"):
            p0_test, cost, it = gfjrsolver.cost_function.inner_solver(init_guess=gfjrsolver.p0_est)
        log(f"[DEBUG] inner FISTA finished: cost={cost} it={it} p0.shape={p0_test.shape}")
        logger.stop_heartbeat()
        logger.close()
        return

    # -------------------------
    # 17) solve
    # -------------------------
    c0 = np.array([4.0, 2.0], dtype=DTYPE) * start
    lower = 0.9 * c0
    upper = 1.1 * c0

    log("START GFJR.solve()")
    with StageTimer("GFJR.solve"):
        gfjrsolver.solve(c0, lower, upper, num_iter_init=num_iter * 2, maxfun=maxfun)

    logger.stop_heartbeat()
    log("ALL DONE")
    logger.close()

if __name__ == "__main__":
    main()