import os
from .fista_tv_overall import fista_tv


def run_fista(
    pmeas, Nx, Ny, Nz,
    forward_op, adjoint_op, roi,
    fista_cfg,
    saving_dir='',
    init_guess=None,
    mpi_rank=0,
    out_print=3
):
    """
    Config-friendly wrapper around fista_tv().

    Parameters
    ----------
    pmeas : array-like
        Measured pressure.
    Nx, Ny, Nz : int
        Volume shape.
    forward_op, adjoint_op : callable
        Forward/adjoint operators.
    roi : ndarray
        ROI mask, must match local shape used by adjoint_op.
    fista_cfg : dict
        Config dict for FISTA, e.g.
        {
            "reg": 1e-4,
            "lip": 5.0,
            "iter": 20,
            "prox_mode": 2,
            "prox_impl": "mix",
            "prox_iter": 50,
            "grad_min": 1e-5,
            "grad_min_init": 1e-4,
            "cost_min": 1e-3,
            "save_freq": 1,
            "use_check": False,
            "check_iter": 10,
            "rel_thr": 1e-3,
            "rel_patience": 2,
            "rel_warmup": 2,
            "div_rel_thr": 1e-2,
            "div_patience": 2,
            "div_warmup": 2
        }
    """
    cfg = dict(fista_cfg or {})

    reg_param = float(cfg.get("reg", 1e-4))
    lip = float(cfg.get("lip", 5.0))
    niter_max = int(cfg.get("iter", 20))
    prox_mode = int(cfg.get("prox_mode", 1))
    prox_impl = str(cfg.get("prox_impl", "mix")).strip().lower()
    prox_iter = int(cfg.get("prox_iter", 50))

    grad_min = cfg.get("grad_min", 1e-5)
    if grad_min is not None:
        grad_min = float(grad_min)

    cost_min = float(cfg.get("cost_min", 1e-3))
    save_freq = int(cfg.get("save_freq", 1))
    use_check = bool(cfg.get("use_check", False))
    check_iter = int(cfg.get("check_iter", 10))

    rel_thr = float(cfg.get("rel_thr", 1e-3))
    rel_patience = int(cfg.get("rel_patience", 2))
    rel_warmup = int(cfg.get("rel_warmup", 2))

    div_rel_thr = float(cfg.get("div_rel_thr", 1e-2))
    div_patience = int(cfg.get("div_patience", 2))
    div_warmup = int(cfg.get("div_warmup", 2))

    positive_constraint = bool(cfg.get("positive_constraint", True))
    enable_tv = bool(cfg.get("enable_tv", True))

    runtime_cfg = dict(cfg.get("runtime", {}) or {})
    worker_script = runtime_cfg.get("worker_script", None)
    runtime_prox_cuda_visible_devices = runtime_cfg.get("prox_cuda_visible_devices", None)
    runtime_prox_nvidia_visible_devices = runtime_cfg.get("prox_nvidia_visible_devices", None)

    binding_cfg = dict(cfg.get("binding", {}) or {})
    prox_gpu_idx = binding_cfg.get("prox_gpu_idx", None)

    # cluster: 优先读环境变量
    prox_cuda_visible_devices = os.environ.get("PROX_CUDA_VISIBLE_DEVICES", "").strip() or None
    prox_nvidia_visible_devices = os.environ.get("PROX_NVIDIA_VISIBLE_DEVICES", "").strip() or None

    # runtime: 次优先读上游传下来的 prox 绑定
    if prox_cuda_visible_devices is None and prox_nvidia_visible_devices is None:
        if runtime_prox_cuda_visible_devices is not None or runtime_prox_nvidia_visible_devices is not None:
            prox_cuda_visible_devices = runtime_prox_cuda_visible_devices
            prox_nvidia_visible_devices = runtime_prox_nvidia_visible_devices

    # binding: 最后回退到 binding.prox_gpu_idx
    if prox_cuda_visible_devices is None and prox_nvidia_visible_devices is None:
        if prox_gpu_idx is not None and str(prox_gpu_idx).strip() != "":
            prox_cuda_visible_devices = str(prox_gpu_idx)
            prox_nvidia_visible_devices = str(prox_gpu_idx)

    return fista_tv(
        pmeas,
        Nx, Ny, Nz,
        forward_op, adjoint_op,
        roi,
        reg_param=reg_param,
        lip=lip,
        positive_constraint=positive_constraint,
        niter_max=niter_max,
        grad_min=grad_min,
        cost_min=cost_min,
        saving_dir=saving_dir,
        prox_mode=prox_mode,
        prox_impl=prox_impl,
        prox_iter=prox_iter,
        out_print=out_print,
        init_guess=init_guess,
        mpi_rank=mpi_rank,
        use_check=use_check,
        check_iter=check_iter,
        save_freq=save_freq,
        enable_tv=enable_tv,
        rel_thr=rel_thr,
        rel_patience=rel_patience,
        rel_warmup=rel_warmup,
        div_rel_thr=div_rel_thr,
        div_patience=div_patience,
        div_warmup=div_warmup,
        worker_script=worker_script,
        prox_cuda_visible_devices=prox_cuda_visible_devices,
        prox_nvidia_visible_devices=prox_nvidia_visible_devices,
    )
