import os
import sys
import importlib
import subprocess
import numpy as np


def _infer_v2_root():
    """
    Infer V2 root from this file location:
      .../new_v2/src/fista_tv_3d_python/prox_dispatch.py
    """
    here = os.path.abspath(os.path.dirname(__file__))
    v2_root = os.path.abspath(os.path.join(here, "..", ".."))
    return v2_root


def _normalize_prox_impl(prox_impl):
    prox_impl = str(prox_impl).strip().lower()
    if prox_impl not in ("mix", "cupy"):
        raise RuntimeError(
            f"Invalid prox_impl={prox_impl} (expected 'mix' or 'cupy')"
        )
    return prox_impl


def call_proximal_impl(prox_impl="mix"):
    """
    Select in-process proximal implementation by explicit impl name only.
    prox_impl:
        - 'mix'
        - 'cupy'
    """
    prox_impl = _normalize_prox_impl(prox_impl)

    if prox_impl == "mix":
        name = "proximal_L_cupy_mix"
    elif prox_impl == "cupy":
        name = "proximal_L_cupy"
    else:
        raise RuntimeError(f"Unsupported prox_impl={prox_impl}")

    pkg = __package__ or "fista_tv_3d_python"
    full_name = f"{pkg}.{name}"
    module = importlib.import_module(full_name)
    return module.proximal_L


def call_prox_subprocess(
    p0_in, Nx, Ny, Nz, reg_eff, positive_constraint, prox_iter,
    saving_dir, iter_idx, mpi_rank, prox_impl,
    worker_script=None,
    prox_cuda_visible_devices=None,
    prox_nvidia_visible_devices=None,
    log_fn=None
):
    """
    Run prox on another GPU via subprocess (UUID-safe, Condor-safe).

    REQUIRE:
      wrapper.sh must export:
        PROX_CUDA_VISIBLE_DEVICES=<GPU-UUID>
        PROX_NVIDIA_VISIBLE_DEVICES=<GPU-UUID>
    """
    prox_impl = _normalize_prox_impl(prox_impl)

    os.makedirs(saving_dir, exist_ok=True)
    in_npy = os.path.join(saving_dir, f"__prox_in_rank{mpi_rank}_iter{iter_idx}.npy")
    out_npy = os.path.join(saving_dir, f"__prox_out_rank{mpi_rank}_iter{iter_idx}.npy")

    np.save(in_npy, np.asarray(p0_in, dtype=np.float32, order="C"))

    if worker_script is not None:
        worker = os.path.abspath(str(worker_script))
    else:
        v2_root = os.environ.get("V2_ROOT")
        if not v2_root:
            v2_root = _infer_v2_root()
        worker = os.path.join(v2_root, "my_code", "run_prox_worker.py")

    if not os.path.exists(worker):
        raise RuntimeError(f"prox worker not found: {worker}")

    cmd = [
        sys.executable, "-u", worker,
        "--in_npy", in_npy,
        "--out_npy", out_npy,
        "--Nx", str(int(Nx)),
        "--Ny", str(int(Ny)),
        "--Nz", str(int(Nz)),
        "--reg", str(float(reg_eff)),
        "--niter", str(int(prox_iter)),
        "--pos", "1" if bool(positive_constraint) else "0",
        "--prox_impl", prox_impl,
    ]

    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    prox_cuda = "" if prox_cuda_visible_devices is None else str(prox_cuda_visible_devices).strip()
    prox_nv = "" if prox_nvidia_visible_devices is None else str(prox_nvidia_visible_devices).strip()

    if not prox_cuda:
        prox_cuda = env.get("PROX_CUDA_VISIBLE_DEVICES", "").strip()
    if not prox_nv:
        prox_nv = env.get("PROX_NVIDIA_VISIBLE_DEVICES", "").strip()

    if not prox_cuda:
        raise RuntimeError(
            "prox CUDA device binding is not set. "
            "Pass prox_cuda_visible_devices explicitly or set PROX_CUDA_VISIBLE_DEVICES."
        )

    env["CUDA_VISIBLE_DEVICES"] = prox_cuda
    env["NVIDIA_VISIBLE_DEVICES"] = prox_nv if prox_nv else prox_cuda
    env["ACC_DEVICE_NUM"] = "0"
    env["PYTHONPATH"] = env.get("PYTHONPATH", "")

    if log_fn is not None:
        log_fn(
            f"Iter {iter_idx} PROX(subprocess) "
            f"prox_impl={prox_impl} "
            f"worker={worker} "
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            f"NVIDIA_VISIBLE_DEVICES={env['NVIDIA_VISIBLE_DEVICES']} "
            f"(explicit_cuda={prox_cuda_visible_devices} explicit_nv={prox_nvidia_visible_devices})",
            mpi_rank
        )

    subprocess.run(cmd, check=True, env=env)

    p0_out = np.load(out_npy).astype(np.float32, copy=False)

    if p0_out.ndim == 1 and p0_out.size == Nx * Ny * Nz:
        p0_out = p0_out.reshape((Nx, Ny, Nz))
    p0_out = np.ascontiguousarray(p0_out, dtype=np.float32)

    try:
        os.remove(in_npy)
        os.remove(out_npy)
    except Exception:
        pass

    return p0_out
