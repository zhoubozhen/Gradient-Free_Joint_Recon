#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

def bind_gpu():
    # 这些 env 是 main 进程在 subprocess.run(env=...) 里传进来的
    prox_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nv_acc = os.environ.get("NV_ACC_DEVICE_NUM", os.environ.get("ACC_DEVICE_NUM", ""))

    print(f"[PROX] CUDA_VISIBLE_DEVICES={prox_cvd}", flush=True)
    print(f"[PROX] NV_ACC_DEVICE_NUM={os.environ.get('NV_ACC_DEVICE_NUM','<unset>')}", flush=True)
    print(f"[PROX] ACC_DEVICE_NUM={os.environ.get('ACC_DEVICE_NUM','<unset>')}", flush=True)

    # 这里不再强制要求 PROX_GPU_PHYSICAL；统一以 CUDA_VISIBLE_DEVICES 为准
    if not prox_cvd:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty in prox worker (expected UUID like GPU-xxxx)")

def import_proximal():
    """
    兼容不同文件名：proximal_L_cupy_mix / proximal_L_cupy
    只要暴露 proximal_L 就行。
    """
    last_err = None
    for modname in [
        "fista_tv_3d_python.proximal_L_cupy_mix",
        "fista_tv_3d_python.proximal_L_cupy",
        "fista_tv_python.proximal_L_cupy_mix",
        "fista_tv_python.proximal_L_cupy",
    ]:
        try:
            m = __import__(modname, fromlist=["proximal_L"])
            prox = getattr(m, "proximal_L")
            return prox, modname
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot import proximal_L from known modules, last_err={last_err}")

def call_proximal(prox_fn, p0_3d, Nx, Ny, Nz, reg, niter, pos):
    """
    不知道你仓库里 proximal_L 的精确签名，所以做多种 fallback。
    """
    # 常见：proximal_L(p0, Nx, Ny, Nz, reg, niter, pos)
    try:
        return prox_fn(p0_3d, Nx, Ny, Nz, reg, niter, pos)
    except TypeError:
        pass

    # 常见：proximal_L(p0, Nx, Ny, Nz, reg_param=..., niter=..., positive_constraint=...)
    try:
        return prox_fn(
            p0_3d, Nx, Ny, Nz,
            reg_param=reg, niter=niter, positive_constraint=pos
        )
    except TypeError:
        pass

    # 常见：proximal_L(p0, reg_param, niter, pos, Nx,Ny,Nz)
    try:
        return prox_fn(p0_3d, reg, niter, pos, Nx, Ny, Nz)
    except TypeError:
        pass

    # 常见：proximal_L(p0, reg_param=..., niter=..., positive_constraint=...)
    try:
        return prox_fn(p0_3d, reg_param=reg, niter=niter, positive_constraint=pos)
    except TypeError as e:
        raise RuntimeError(f"proximal_L signature mismatch: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npy", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--Nx", type=int, required=True)
    ap.add_argument("--Ny", type=int, required=True)
    ap.add_argument("--Nz", type=int, required=True)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--niter", type=int, required=True)
    ap.add_argument("--pos", type=int, default=1)
    args = ap.parse_args()

    bind_gpu()

    # import cupy AFTER CUDA_VISIBLE_DEVICES is set (already set by env)
    import cupy as cp
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else str(props["name"])
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[PROX] cupy device id={dev.id} name={name} mem_free/total={free/1e9:.2f}/{total/1e9:.2f}GB", flush=True)

    x = np.load(args.in_npy, mmap_mode=None)
    x = np.asarray(x, dtype=np.float32)

    # 输入是 1D（fista_tv_overall.py 保存的是 flat）
    if x.ndim == 1:
        if x.size != args.Nx * args.Ny * args.Nz:
            raise RuntimeError(f"in_npy size mismatch: got {x.size}, expected {args.Nx*args.Ny*args.Nz}")
        x3 = x.reshape(args.Nx, args.Ny, args.Nz)
    elif x.ndim == 3:
        x3 = x
    else:
        raise RuntimeError(f"Unexpected in_npy ndim={x.ndim}")

    prox_fn, modname = import_proximal()
    print(f"[PROX] using proximal_L from {modname}", flush=True)

    y3 = call_proximal(
        prox_fn,
        x3,
        args.Nx, args.Ny, args.Nz,
        args.reg, args.niter,
        int(args.pos),
    )

    # y3 可能是 numpy 或 cupy
    if hasattr(y3, "get"):  # cupy array
        y3 = y3.get()
    y3 = np.asarray(y3, dtype=np.float32)

    # 保存成 flat，和主进程读取方式一致
    np.save(args.out_npy, y3.reshape(-1))
    print(f"[PROX] wrote out_npy={args.out_npy} shape={y3.shape}", flush=True)

if __name__ == "__main__":
    main()
