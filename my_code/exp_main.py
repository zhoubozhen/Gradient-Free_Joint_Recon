#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

from tranPACT import GFJRSolver, OptimParam
from exp_util import (
    load_json,
    load_exp_data_from_config,
    setup_wave_solver_from_config,
    build_saving_dir,
)


def setup_env(cfg):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    binding = dict(cfg.get("binding", {}) or {})
    main_gpu = binding.get("main_gpu_idx", None)

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "" and main_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(main_gpu)
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(main_gpu))

    os.environ.setdefault("NV_ACC_DEVICE_TYPE", "NVIDIA")
    os.environ.setdefault("NV_ACC_DEVICE_NUM", "0")
    os.environ.setdefault("ACC_DEVICE_NUM", "0")


def setup_devito(cfg):
    from devito import configuration
    import devito.arch.compiler as devito_compiler
    import codepy.toolchain as codepy_toolchain

    configuration["language"] = "openacc"
    configuration["platform"] = "nvidiaX"
    configuration["compiler"] = "nvc"
    configuration["log-level"] = cfg.get("devito_log_level", "INFO")
    configuration["ignore-unknowns"] = 1

    compiler = configuration["compiler"]

    bad_flags = {"-Wno-unused-result", "-ffast-math"}
    if hasattr(compiler, "cflags"):
        compiler.cflags = [x for x in compiler.cflags if x not in bad_flags]
    if hasattr(compiler, "ldflags"):
        compiler.ldflags = [x for x in compiler.ldflags if x not in bad_flags]

    # cluster 某些节点 `nvc++ --version` 会报 Unsupported processor
    try:
        compiler.get_version = lambda: "nvc-cluster-fixed"
    except Exception:
        pass
    try:
        type(compiler).get_version = lambda self: "nvc-cluster-fixed"
    except Exception:
        pass
    try:
        devito_compiler.Compiler.get_version = lambda self: "nvc-cluster-fixed"
    except Exception:
        pass
    try:
        codepy_toolchain.GCCToolchain.get_version = lambda self: "nvc-cluster-fixed"
    except Exception:
        pass

    print("[DEVITO] compiler =", compiler, flush=True)
    if hasattr(compiler, "cflags"):
        print("[DEVITO] cflags   =", compiler.cflags, flush=True)
    if hasattr(compiler, "ldflags"):
        print("[DEVITO] ldflags  =", compiler.ldflags, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)

    setup_env(cfg)
    setup_devito(cfg)

    print("==== LOAD EXP DATA ====", flush=True)
    data = load_exp_data_from_config(cfg)

    rec_pos = data["rec_pos"]
    forward = data["forward"]
    Nt = data["Nt"]
    fn = data["fn"]

    print(f"[DATA] rec_pos={rec_pos.shape}", flush=True)
    print(f"[DATA] forward={forward.shape}", flush=True)
    print(f"[DATA] Nt={Nt}", flush=True)

    print("==== BUILD SOLVER ====", flush=True)
    solver, model, cl_start, cs_start, fn, skull_type = \
        setup_wave_solver_from_config(cfg, data)

    saving_dir = build_saving_dir(cfg, skull_type)
    print(f"[SAVE] {saving_dir}", flush=True)

    with open(os.path.join(saving_dir, "config_record.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    fista_cfg = dict(cfg.get("fista", {}) or {})

    opt_param = OptimParam(
        reg=float(fista_cfg.get("reg", 0.0)),
        num_iter=int(fista_cfg.get("iter", 20)),
        out_print=int(cfg.get("out_print", 3)),
        lip=float(fista_cfg.get("lip", 5.0)),
        saving_dir=saving_dir,
        use_check=bool(fista_cfg.get("use_check", True)),
        save_freq=int(fista_cfg.get("save_freq", 1)),
    )

    opt_param.prox_mode = int(fista_cfg.get("prox_mode", 1))
    opt_param.prox_impl = str(fista_cfg.get("prox_impl", "mix"))
    opt_param.prox_iter = int(fista_cfg.get("prox_iter", 50))

    opt_param.grad_min = float(fista_cfg.get("grad_min", 1e-5))
    opt_param.cost_min = float(fista_cfg.get("cost_min", 1e-3))
    opt_param.check_iter = int(fista_cfg.get("check_iter", 1))

    opt_param.rel_thr = float(fista_cfg.get("rel_thr", 1e-2))
    opt_param.rel_patience = int(fista_cfg.get("rel_patience", 2))
    opt_param.rel_warmup = int(fista_cfg.get("rel_warmup", 2))

    opt_param.div_rel_thr = float(fista_cfg.get("div_rel_thr", 1e-2))
    opt_param.div_patience = int(fista_cfg.get("div_patience", 2))
    opt_param.div_warmup = int(fista_cfg.get("div_warmup", 2))

    runtime_cfg = dict(fista_cfg.get("runtime", {}) or {})
    worker_script = runtime_cfg.get("worker_script", None)
    opt_param.worker_script = os.path.expandvars(worker_script) if worker_script else None

    binding = dict(cfg.get("binding", {}) or {})
    prox_gpu_idx = binding.get("prox_gpu_idx", None)
    if prox_gpu_idx is not None:
        opt_param.prox_cuda_visible_devices = str(prox_gpu_idx)
        opt_param.prox_nvidia_visible_devices = str(prox_gpu_idx)

    gfjrsolver = GFJRSolver(
        solver=solver,
        measured_pressure=forward.ravel().copy(),
        fn=fn,
        opt_param=opt_param,
        comm=None,
        Nt=Nt,
    )

    start = float(cfg.get("start", 1.0))
    maxfun = int(cfg.get("maxfun", 60))

    if skull_type == "aubry":
        c0 = np.array([4.0, 2.0], dtype=np.float32) * start
    else:
        c0 = np.array([2.898, 1.4], dtype=np.float32) * start

    lower = 0.85 * c0
    upper = 1.15 * c0

    print("==== START GFJR ====", flush=True)
    gfjrsolver.solve(c0, lower, upper, num_iter_init=opt_param.num_iter * 2, maxfun=maxfun, use_static=False)
    print("==== DONE ====", flush=True)


if __name__ == "__main__":
    main()
