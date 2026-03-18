# fista_tv_overall.py  (FULL MODIFIED VERSION + DIVERGENCE STOP)
import os
import sys
import math
import numpy as np
import scipy.io as sio
import time
import importlib
import subprocess

from .cost_func_tv import cost_func_tv
from .prox_dispatch import call_proximal_impl, call_prox_subprocess

# ==========================================================
# >>>>>>>>>>>>>>>>>>>> TV GLOBAL SWITCH <<<<<<<<<<<<<<<<<<<<
# ==========================================================
ENABLE_TV = True        # <<< True: TV prox ON | False: TV prox OFF (fake-prox)
# ==========================================================

# ==========================================================
# logging knobs (only affect prints, no algorithm changes)
# ==========================================================
LOG_EVERY_ITER = True          # print one summary line every FISTA iter
LOG_DP0_NORM = True            # print ||dp0||_2 summary (cheap)
LOG_COST_TOL = True            # print early-stop tol & counters

# ==========================================================
# NEW: relative convergence early-stop (SAFE, does NOT change math)
# ==========================================================
REL_EARLY_STOP = True          # True: enable rel-based early stop
REL_THR_DEFAULT = 1e-3         # stop if rel < thr ...
REL_PATIENCE_DEFAULT = 2       # ... for this many consecutive iters
REL_WARMUP_DEFAULT = 2         # don't count rel stop before iter>=2

# ==========================================================
# NEW: divergence early-stop (dcost + rel)
# ==========================================================
DIV_EARLY_STOP = True          # True: enable divergence stop
DIV_REL_THR_DEFAULT = 1e-2     # treat as divergence if (dcost>0 and rel>thr)
DIV_PATIENCE_DEFAULT = 2       # ... for this many consecutive iters
DIV_WARMUP_DEFAULT = 2         # don't count divergence stop before iter>=2
# ==========================================================

# ==========================================================
# utils: timing + memory
# ==========================================================
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _mem():
    try:
        out = {}
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:", "VmSize:")):
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
        return f"RSS={out.get('VmRSS','?')} Swap={out.get('VmSwap','?')} HWM={out.get('VmHWM','?')}"
    except Exception:
        return "RSS=? Swap=?"

def _log(msg, rank=0):
    if rank == 0:
        print(f"[{_now()}][FISTA] {msg} | {_mem()}", flush=True)

def _fmt(x):
    try:
        return f"{float(x):.6e}"
    except Exception:
        return str(x)


# ------------------------------
# NEW: subprocess prox launcher (used ONLY when prox_mode==2)
# ------------------------------


def readCheck(fname, niter_max):
    checkpoint = sio.loadmat(fname)
    start_iter = checkpoint['iter'][0][0] + 1
    tp = checkpoint['tp'][0][0]
    p0y = checkpoint['p0y']
    p0p = checkpoint['p0p']
    cost = checkpoint['cost']
    if cost.shape[0] < niter_max:
        old_cost = cost.copy()
        cost = np.zeros((niter_max, 1))
        cost[:old_cost.shape[0], 0] = old_cost[:, 0]
    return start_iter, tp, p0y, p0p, cost

def initialize(init_guess, shape, niter_max):
    start_iter = 1
    tp = 1
    if init_guess is not None:
        p0y = init_guess.copy().reshape(shape)
        p0p = init_guess.copy().reshape(shape)
    else:
        p0p = np.zeros(shape, dtype=np.float32)
        p0y = np.zeros(shape, dtype=np.float32)
    cost = np.zeros((niter_max, 1))
    return start_iter, tp, p0y, p0p, cost

# ==========================================================
# FISTA-TV (TV ENABLE / DISABLE SWITCHABLE VERSION)
# ==========================================================
def fista_tv(pmeas, Nx, Ny, Nz,
             forward_op, adjoint_op,
             roi, reg_param, lip,
             positive_constraint, niter_max,
             grad_min=None, cost_min=1e-8,
             saving_dir='', prox_mode=1, prox_impl='mix',
             prox_iter=50, out_print=3,
             init_guess=None, mpi_rank=0,
             use_check=False, check_iter=5,
             save_freq=1,
             enable_tv=ENABLE_TV,
             log_every_iter=LOG_EVERY_ITER,
             log_dp0_norm=LOG_DP0_NORM,
             log_cost_tol=LOG_COST_TOL,
             rel_early_stop=REL_EARLY_STOP,
             div_early_stop=DIV_EARLY_STOP,
             # NEW (optional): rel-based stop controls
             rel_thr=REL_THR_DEFAULT,
             rel_patience=REL_PATIENCE_DEFAULT,
             rel_warmup=REL_WARMUP_DEFAULT,
             # NEW (optional): divergence stop controls
             div_rel_thr=DIV_REL_THR_DEFAULT,
             div_patience=DIV_PATIENCE_DEFAULT,
             div_warmup=DIV_WARMUP_DEFAULT,
             worker_script=None,
             prox_cuda_visible_devices=None,
             prox_nvidia_visible_devices=None,
             **kwargs):

    # ------------------------------------------------------
    # ENTRY LOG
    # ------------------------------------------------------
    mode = "TV ENABLED" if enable_tv else "TV DISABLED"
    _log(f"ENTER fista_tv | {mode}", mpi_rank)

    # --- summarize run config for convergence judging ---
    try:
        roi_nnz = int(np.count_nonzero(roi))
        roi_tot = int(roi.size)
        roi_ratio = roi_nnz / max(1, roi_tot)
    except Exception:
        roi_nnz, roi_tot, roi_ratio = -1, -1, float("nan")

    _log(
        "CONFIG "
        f"niter_max={niter_max} reg={reg_param} lip={lip} "
        f"prox_iter={prox_iter} prox_mode={prox_mode} prox_impl={prox_impl} "
        f"positive={bool(positive_constraint)} "
        f"roi_shape={getattr(roi,'shape',None)} roi_nnz={roi_nnz} ({roi_ratio:.3e}) "
        f"Nxyz=({Nx},{Ny},{Nz}) "
        f"use_check={bool(use_check)} save_freq={save_freq} "
        f"REL_EARLY_STOP={bool(rel_early_stop)} rel_thr={rel_thr} rel_patience={rel_patience} rel_warmup={rel_warmup} "
        f"DIV_EARLY_STOP={bool(div_early_stop)} div_rel_thr={div_rel_thr} div_patience={div_patience} div_warmup={div_warmup} "
        f"worker_script={worker_script} "
        f"prox_cuda_visible_devices={prox_cuda_visible_devices} "
        f"prox_nvidia_visible_devices={prox_nvidia_visible_devices}",
        mpi_rank
    )

    # allocate
    pest = np.zeros_like(pmeas, dtype=np.float32)

    fname = os.path.join(saving_dir, 'checkpoint.mat')
    if not os.path.exists(fname):
        use_check = False

    if use_check:
        start_iter, tp, p0y, p0p, cost = readCheck(fname, niter_max)
        _log(f"CHECKPOINT loaded: start_iter={start_iter} tp={tp}", mpi_rank)
    else:
        start_iter, tp, p0y, p0p, cost = initialize(init_guess, roi.shape, niter_max)
        _log(f"INIT: start_iter={start_iter} tp={tp} init_guess={'YES' if init_guess is not None else 'NO'}", mpi_rank)

    # for convergence / early-stop
    costp = 1e100
    cost_prev_for_log = None

    # relative early-stop counters
    rel_iter = 0

    # divergence early-stop counters
    div_iter = 0

    p0_out = p0p.copy()
    min_iter = 0
    grad_iter = 0

    proximal_L = None
    if enable_tv:
        pmode = int(prox_mode) if not isinstance(prox_mode, (tuple, list)) else int(prox_mode[0])
        if pmode == 2:
            _log(
                f"proximal selected: prox_mode==2 -> subprocess worker (GPU split), prox_impl={prox_impl}",
                mpi_rank
            )
        else:
            proximal_L = call_proximal_impl(prox_impl)
            _log(
                f"proximal selected: prox_mode={pmode} prox_impl={prox_impl} -> "
                f"{proximal_L.__module__}.{proximal_L.__name__}",
                mpi_rank
            )

    if log_cost_tol:
        _log(
            f"EARLY-STOP thresholds: "
            f"cost_min={cost_min} (need >20 consecutive hits), "
            f"grad_min={grad_min}, "
            f"rel_thr={rel_thr} rel_patience={rel_patience}, "
            f"div_rel_thr={div_rel_thr} div_patience={div_patience}",
            mpi_rank
        )

    # ======================================================
    # FISTA LOOP
    # ======================================================
    for iter in range(start_iter, niter_max + 1):

        t_iter0 = time.time()
        _log(f"Iter {iter} ENTER", mpi_rank)

        # ---------------- forward ----------------
        if iter > 1 or (init_guess is not None and np.any(init_guess)):
            _log(f"Iter {iter} BEFORE forward", mpi_rank)
            pest = forward_op(p0y[roi > 0])
            _log(
                f"Iter {iter} AFTER  forward | shape={getattr(pest,'shape',None)} dtype={getattr(pest,'dtype',None)}",
                mpi_rank
            )
        else:
            _log(f"Iter {iter} SKIP forward (iter==1 and init_guess empty)", mpi_rank)

        # ---------------- cost ----------------
        _log(f"Iter {iter} BEFORE cost_func_tv", mpi_rank)
        costn = cost_func_tv(pmeas, pest, reg_param, p0y, *p0y.shape)
        cost[iter - 1] = costn

        # ---- convergence metrics for logs ----
        if cost_prev_for_log is None:
            dcost = float("nan")
            rel = float("nan")
        else:
            dcost = float(costn) - float(cost_prev_for_log)
            rel = abs(dcost) / (abs(float(cost_prev_for_log)) + 1e-12)

        _log(
            f"Iter {iter} AFTER  cost_func_tv "
            f"cost={float(costn):.6g} dcost={dcost:.3e} rel={rel:.3e} "
            f"(reg={reg_param} lip={lip})",
            mpi_rank
        )

        # ---------------- NEW early stop (divergence) ----------------
        # 当 cost 持续上升并且相对变化很大时，认为发散：立刻刹车
        if div_early_stop and (iter >= max(start_iter, div_warmup)) and (not np.isnan(rel)) and (not np.isnan(dcost)):
            if (dcost > 0) and (rel > float(div_rel_thr)):
                div_iter += 1
            else:
                div_iter = 0

            _log(
                f"Iter {iter} DIV_STOP counter={div_iter}/{div_patience} "
                f"(dcost={dcost:.3e} rel={rel:.3e} thr={div_rel_thr})",
                mpi_rank
            )

            if div_iter >= int(div_patience):
                _log(
                    f"Iter {iter} EARLY STOP (divergence) | "
                    f"dcost={dcost:.3e} > 0 and rel={rel:.3e} > {float(div_rel_thr):.3e} "
                    f"for {int(div_patience)} iters",
                    mpi_rank
                )
                break

        # store for next iter log
        cost_prev_for_log = costn

        # ---------------- residual ----------------
        pest = pest - pmeas

        # ---------------- adjoint ----------------
        _log(f"Iter {iter} BEFORE adjoint", mpi_rank)
        dp0 = adjoint_op(pest)
        _log(
            f"Iter {iter} AFTER  adjoint | shape={getattr(dp0,'shape',None)} dtype={getattr(dp0,'dtype',None)}",
            mpi_rank
        )

        if dp0.shape != roi.shape:
            raise RuntimeError(
                f"MPI adjoint returned local shape {dp0.shape}, "
                f"but ROI is {roi.shape}."
            )

        # optional dp0 norm (cheap)
        if log_dp0_norm:
            try:
                dp0_norm = float(np.linalg.norm(dp0.ravel()))
                _log(f"Iter {iter} dp0_norm2={dp0_norm:.6e}", mpi_rank)
            except Exception as e:
                _log(f"Iter {iter} dp0_norm2=NA ({e})", mpi_rank)

        dp0[roi == 0] = 0

        # ==================================================
        # PROXIMAL STEP
        # ==================================================
        if enable_tv:
            _log(f"Iter {iter} ENTER TV proximal", mpi_rank)
            t0 = time.time()

            # ---- IMPORTANT: prox_mode branching ----
            pmode = int(prox_mode) if not isinstance(prox_mode, (tuple, list)) else int(prox_mode[0])

            if pmode == 1:
                # keep original logic (in-proc prox)
                p0 = proximal_L(
                    p0y - (2.0 / lip) * dp0,
                    Nx, Ny, Nz,
                    2.0 * reg_param / lip,
                    positive_constraint,
                    prox_iter
                )

            elif pmode == 2:
                # subprocess prox (GPU split)
                p0 = call_prox_subprocess(
                    p0y - (2.0 / lip) * dp0,
                    Nx, Ny, Nz,
                    2.0 * reg_param / lip,
                    positive_constraint,
                    prox_iter,
                    saving_dir=saving_dir,
                    iter_idx=iter,
                    mpi_rank=mpi_rank,
                    prox_impl=prox_impl,
                    worker_script=worker_script,
                    prox_cuda_visible_devices=prox_cuda_visible_devices,
                    prox_nvidia_visible_devices=prox_nvidia_visible_devices
                )
            else:
                # fallback: keep original logic for other modes
                if proximal_L is None:
                    proximal_L = call_proximal_impl(prox_impl)
                p0 = proximal_L(
                    p0y - (2.0 / lip) * dp0,
                    Nx, Ny, Nz,
                    2.0 * reg_param / lip,
                    positive_constraint,
                    prox_iter
                )

            _log(
                f"Iter {iter} EXIT  TV proximal dt={time.time()-t0:.3f}s | p0 dtype={getattr(p0,'dtype',None)}",
                mpi_rank
            )

        else:
            _log(f"Iter {iter} ENTER fake-prox (TV disabled)", mpi_rank)
            t0 = time.time()

            p0 = p0y - (2.0 / lip) * dp0
            if positive_constraint:
                p0[p0 < 0] = 0

            _log(f"Iter {iter} EXIT  fake-prox dt={time.time()-t0:.3f}s", mpi_rank)

        # ---------------- save ----------------
        if mpi_rank == 0 and save_freq > 0 and iter % save_freq == 0:
            try:
                p0.astype(np.float32).ravel().tofile(
                    os.path.join(saving_dir, f'p0_iter_{iter}.DAT')
                )
                _log(f"Iter {iter} SAVE p0_iter_{iter}.DAT", mpi_rank)
            except Exception as e:
                _log(f"Iter {iter} SAVE failed: {e}", mpi_rank)

        # ---------------- FISTA update ----------------
        tn = (1 + math.sqrt(1 + 4 * tp ** 2)) / 2
        p0y = p0 + ((tp - 1) / tn) * (p0 - p0p)

        # ---------------- original early stop (cost) ----------------
        # (kept EXACTLY as you had)
        if (float(costp) - float(costn)) < cost_min:
            min_iter += 1
        else:
            costp = costn
            p0_out = p0
            min_iter = 0

        # ---------------- NEW early stop (relative) ----------------
        if rel_early_stop and (iter >= max(start_iter, rel_warmup)) and (not np.isnan(rel)):
            if rel < float(rel_thr):
                rel_iter += 1
            else:
                rel_iter = 0

            _log(f"Iter {iter} REL_STOP counter={rel_iter}/{rel_patience} (rel={rel:.3e} thr={rel_thr})", mpi_rank)

            if rel_iter >= int(rel_patience):
                _log(
                    f"Iter {iter} EARLY STOP (rel) | "
                    f"rel={rel:.3e} < {float(rel_thr):.3e} for {int(rel_patience)} iters",
                    mpi_rank
                )
                break

        if log_every_iter:
            _log(
                f"Iter {iter} SUMMARY "
                f"cost={float(costn):.6e} "
                f"best_cost={float(costp):.6e} "
                f"rel={rel:.3e} "
                f"min_iter={min_iter}/20 "
                f"reg={reg_param} lip={lip}",
                mpi_rank
            )

        if min_iter > 20:
            _log(f"Iter {iter} EARLY STOP (cost) | min_iter={min_iter} cost_min={cost_min}", mpi_rank)
            break

        # ---------------- early stop (grad) ----------------
        if grad_min is not None:
            grad_ratio = np.sum((p0 - p0p) ** 2) / (np.sum(p0p ** 2) + 1e-12) * 1e2
            if grad_ratio < grad_min:
                grad_iter += 1
            else:
                grad_iter = 0

            _log(f"Iter {iter} grad_ratio={grad_ratio:.3e} grad_iter={grad_iter}/3 (grad_min={grad_min})", mpi_rank)

        if grad_iter > 3:
            _log(f"Iter {iter} EARLY STOP (grad) | grad_iter={grad_iter} grad_min={grad_min}", mpi_rank)
            break

        tp = tn
        p0p = p0

        _log(f"Iter {iter} EXIT total_dt={time.time()-t_iter0:.3f}s", mpi_rank)

    _log(f"EXIT fista_tv | {mode} | iters_done={iter} | final_best_cost={float(costp):.6e}", mpi_rank)
    return p0_out, costp, iter
