#!/usr/bin/env python3
"""GFJR helper utilities extracted from gfjr_stable.py.

Keep this module import-safe: no cupy/devito import at module import time.
"""

from __future__ import annotations

import os
import time
import threading
import subprocess
from datetime import datetime
import math
from typing import Optional, Tuple

import numpy as np


def print_main_env_snapshot(tag: str = "MAIN") -> None:
    print(f"==== [{tag}] ENV SNAPSHOT ====", flush=True)
    print(f"[{tag}] HOST =", os.uname().nodename, flush=True)
    print(f"[{tag}] CUDA_DEVICE_ORDER =", os.environ.get("CUDA_DEVICE_ORDER", "<unset>"), flush=True)
    print(f"[{tag}] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)
    print(f"[{tag}] NVIDIA_VISIBLE_DEVICES =", os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>"), flush=True)
    print(f"[{tag}] _CONDOR_AssignedGPUs =", os.environ.get("_CONDOR_AssignedGPUs", "<unset>"), flush=True)
    print(f"[{tag}] _CONDOR_SLOT =", os.environ.get("_CONDOR_SLOT", "<unset>"), flush=True)
    print(f"[{tag}] NV_ACC_DEVICE_NUM =", os.environ.get("NV_ACC_DEVICE_NUM", "<unset>"), flush=True)
    print(f"[{tag}] ACC_DEVICE_NUM =", os.environ.get("ACC_DEVICE_NUM", "<unset>"), flush=True)
    print("=============================", flush=True)


def try_print_nvidia_smi(tag: str = "MAIN") -> None:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
        print(f"==== [{tag}] nvidia-smi -L ====", flush=True)
        print(out, flush=True)
        print("==============================", flush=True)
    except Exception as e:
        print(f"[{tag}] nvidia-smi -L failed: {e}", flush=True)


def _read_proc_status() -> dict:
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


def _gpu_snapshot_enabled(log_gpu: bool) -> str:
    if not log_gpu:
        return ""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        s = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        gpu_strs = []
        for line in s.splitlines():
            gid, util, mem_used, mem_total = line.split(",")
            mem_used_gb = int(mem_used) / 1024
            mem_total_gb = int(mem_total) / 1024
            gpu_strs.append(
                f"{gid.strip()}: util={util.strip()}% mem={mem_used_gb:.1f}/{mem_total_gb:.1f}GB"
            )
        return " | GPU: " + " ; ".join(gpu_strs)
    except Exception:
        return ""


def _kb_to_gb(s: str) -> str:
    try:
        return f"{int(s.split()[0]) / (1024**2):.2f} GB"
    except Exception:
        return s


class Logger:
    """Lightweight logger with optional GPU+memory snapshot and heartbeat."""

    def __init__(self, rank: int = 0, size: int = 1, log_gpu: bool = True, heartbeat_sec: int = 60):
        self.rank = int(rank)
        self.size = int(size)
        self.log_gpu = bool(log_gpu)
        self.heartbeat_sec = int(heartbeat_sec)
        self._fh = None  # type: ignore
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

    def set_log_file(self, path: str) -> None:
        # line-buffered
        self._fh = open(path, "w", buffering=1)

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        finally:
            self._fh = None

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, msg: str) -> None:
        prefix = f"[{self._now()}][rank {self.rank}/{self.size}][pid {os.getpid()}]"
        st = _read_proc_status()
        mem = ""
        if st:
            mem = (
                f" | RSS={_kb_to_gb(st.get('VmRSS','?'))}"
                f" Swap={_kb_to_gb(st.get('VmSwap','?'))}"
                f" HWM={_kb_to_gb(st.get('VmHWM','?'))}"
            )
        g = _gpu_snapshot_enabled(self.log_gpu)
        line = f"{prefix} {msg}{mem}{g}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def stage(self, name: str) -> "StageTimer":
        return StageTimer(self, name)

    def start_heartbeat(self, tag: str = "HEARTBEAT (alive)") -> None:
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return

        self._hb_stop.clear()

        def loop():
            while not self._hb_stop.is_set():
                self.log(tag)
                self._hb_stop.wait(self.heartbeat_sec)

        self._hb_thread = threading.Thread(target=loop, daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        self._hb_stop.set()


class StageTimer:
    def __init__(self, logger: Logger, name: str):
        self.logger = logger
        self.name = name
        self.t0: Optional[float] = None

    def __enter__(self):
        self.t0 = time.time()
        self.logger.log(f"--> ENTER {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - (self.t0 or time.time())
        if exc is None:
            self.logger.log(f"<-- EXIT  {self.name}  dt={dt:.3f}s")
        else:
            self.logger.log(f"<!! EXC   {self.name}  dt={dt:.3f}s  exc={exc_type.__name__}: {exc}")
        return False


def downsample_index(n: int, stride: float) -> np.ndarray:
    """Nearest index downsample that supports float stride."""
    stride = float(stride)
    if stride <= 1.0:
        return np.arange(n, dtype=np.int64)
    n_new = int(math.ceil(n / stride))
    idx = np.round(np.linspace(0, n - 1, n_new)).astype(np.int64)
    idx = np.unique(idx)
    return idx


def apply_downsample_3d(arr: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    return arr[np.ix_(ix, iy, iz)]
