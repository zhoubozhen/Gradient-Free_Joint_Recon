#!/usr/bin/env python3
import os
import sys
import argparse
import time
import threading
import subprocess
import numpy as np

this_file = os.path.abspath(__file__)
# run_prox_worker.py 在 new_v2/my_code/，所以 src 在 ../src
src_dir = os.path.abspath(os.path.join(os.path.dirname(this_file), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
print(f"[PROX] sys.path[0]={sys.path[0]}", flush=True)


# -------------------------
# Heartbeat helpers
# -------------------------
def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _get_rank_world():
    r = os.environ.get("OMPI_COMM_WORLD_RANK")
    n = os.environ.get("OMPI_COMM_WORLD_SIZE")
    if r is None:
        r = os.environ.get("PMI_RANK", "0")
    if n is None:
        n = os.environ.get("PMI_SIZE", "1")
    try:
        r = int(r)
    except Exception:
        r = 0
    try:
        n = int(n)
    except Exception:
        n = 1
    return r, n


def _read_proc_status_kb(key: str):
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])  # kB
    except Exception:
        pass
    return None


def _mem_snapshot_gb():
    rss_kb = None
    swap_kb = None
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        rss_kb = int(p.memory_info().rss / 1024)
    except Exception:
        rss_kb = _read_proc_status_kb("VmRSS")

    swap_kb = _read_proc_status_kb("VmSwap")
    hwm_kb = _read_proc_status_kb("VmHWM")

    def kb_to_gb(x):
        return 0.0 if x is None else (x / 1024.0 / 1024.0)

    return kb_to_gb(rss_kb), kb_to_gb(swap_kb), kb_to_gb(hwm_kb)


def _gpu_snapshot_nvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            GiB = 1024.0 ** 3
            used_gb = mem.used / GiB
            total_gb = mem.total / GiB
            out.append({
                "idx": i,
                "util": int(util),
                "used_gb": used_gb,
                "total_gb": total_gb
            })
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return None


def _gpu_snapshot_smi():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = [x.strip() for x in p.stdout.strip().splitlines() if x.strip()]
        out = []
        for i, line in enumerate(lines):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                util = int(float(parts[0]))
                used_mib = float(parts[1])
                total_mib = float(parts[2])
                out.append({
                    "idx": i,
                    "util": util,
                    "used_gb": used_mib / 1024.0,
                    "total_gb": total_mib / 1024.0,
                })
        return out
    except Exception:
        return None


def _gpu_snapshot():
    snap = _gpu_snapshot_nvml()
    if snap is not None:
        return snap
    snap = _gpu_snapshot_smi()
    if snap is not None:
        return snap
    return []


def _format_gpu_line(gpus):
    if not gpus:
        return "GPU: <unavailable>"
    chunks = []
    for g in gpus:
        chunks.append(f"{g['idx']}: util={g['util']}% mem={g['used_gb']:.1f}/{g['total_gb']:.1f}GB")
    return "GPU: " + " ; ".join(chunks)


def start_heartbeat(tag="[PROX] HEARTBEAT", interval_sec=10):
    stop_evt = threading.Event()
    rank, world = _get_rank_world()
    pid = os.getpid()

    def loop():
        while not stop_evt.is_set():
            rss, swp, hwm = _mem_snapshot_gb()
            gpus = _gpu_snapshot()
            ts = _now_str()
            print(
                f"[{ts}][rank {rank}/{world}][pid {pid}] {tag} (alive) | "
                f"RSS={rss:.2f} GB Swap={swp:.2f} GB HWM={hwm:.2f} GB | "
                f"{_format_gpu_line(gpus)}",
                flush=True,
            )
            stop_evt.wait(interval_sec)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return stop_evt


# -------------------------
# Worker logic
# -------------------------
def bind_gpu():
    prox_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[PROX] CUDA_VISIBLE_DEVICES={prox_cvd}", flush=True)
    print(f"[PROX] NV_ACC_DEVICE_NUM={os.environ.get('NV_ACC_DEVICE_NUM','<unset>')}", flush=True)
    print(f"[PROX] ACC_DEVICE_NUM={os.environ.get('ACC_DEVICE_NUM','<unset>')}", flush=True)

    if not prox_cvd:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty in prox worker (expected UUID like GPU-xxxx)")


def import_proximal(prox_impl: str):
    prox_impl = str(prox_impl).strip().lower()

    if prox_impl == "mix":
        cand = [
            "fista_tv_3d_python.proximal_L_cupy_mix",
            "fista_tv_python.proximal_L_cupy_mix",
        ]
    elif prox_impl == "cupy":
        cand = [
            "fista_tv_3d_python.proximal_L_cupy",
            "fista_tv_python.proximal_L_cupy",
        ]
    else:
        raise RuntimeError(f"Unsupported prox_impl={prox_impl} (expected 'mix' or 'cupy')")

    last_err = None
    for modname in cand:
        try:
            m = __import__(modname, fromlist=["proximal_L"])
            prox = getattr(m, "proximal_L")
            return prox, modname
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Cannot import proximal_L for prox_impl={prox_impl}, last_err={last_err}"
    )


def call_proximal(prox_fn, p0_3d, Nx, Ny, Nz, reg, niter, pos):
    try:
        return prox_fn(p0_3d, Nx, Ny, Nz, reg, niter, pos)
    except TypeError:
        pass
    try:
        return prox_fn(p0_3d, Nx, Ny, Nz, reg_param=reg, niter=niter, positive_constraint=pos)
    except TypeError:
        pass
    try:
        return prox_fn(p0_3d, reg, niter, pos, Nx, Ny, Nz)
    except TypeError:
        pass
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
    ap.add_argument("--prox_impl", type=str, required=True, choices=["mix", "cupy"])
    args = ap.parse_args()

    bind_gpu()

    hb_sec = int(os.environ.get("PROX_HEARTBEAT_SEC", "10"))
    hb_stop = start_heartbeat(tag="HEARTBEAT", interval_sec=hb_sec)

    try:
        import cupy as cp
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else str(props["name"])
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"[PROX] cupy device id={dev.id} name={name} mem_free/total={free/1e9:.2f}/{total/1e9:.2f}GB", flush=True)

        x = np.load(args.in_npy, mmap_mode=None)
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            expected = args.Nx * args.Ny * args.Nz
            if x.size != expected:
                raise RuntimeError(f"in_npy size mismatch: got {x.size}, expected {expected}")
            x3 = x.reshape(args.Nx, args.Ny, args.Nz)
        elif x.ndim == 3:
            x3 = x
        else:
            raise RuntimeError(f"Unexpected in_npy ndim={x.ndim}")

        print(f"[PROX] prox_impl={args.prox_impl}", flush=True)
        prox_fn, modname = import_proximal(args.prox_impl)
        print(f"[PROX] using proximal_L from {modname}", flush=True)

        y3 = call_proximal(
            prox_fn,
            x3,
            args.Nx, args.Ny, args.Nz,
            args.reg, args.niter,
            int(args.pos),
        )

        if hasattr(y3, "get"):
            y3 = y3.get()
        y3 = np.asarray(y3, dtype=np.float32)

        np.save(args.out_npy, y3.reshape(-1))
        print(f"[PROX] wrote out_npy={args.out_npy} shape={y3.shape}", flush=True)

    finally:
        hb_stop.set()


if __name__ == "__main__":
    main()