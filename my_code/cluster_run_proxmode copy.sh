#!/usr/bin/env bash
set -eo pipefail
set -x

############################################################
# 0. Condor execute node PATH 修复（只补，不覆盖）
############################################################
if ! command -v dirname >/dev/null 2>&1; then
  export PATH="/usr/bin:/bin:/usr/local/bin:${PATH:-}"
fi

############################################################
# 1. 参数
############################################################
ENV_NAME=devito

# 角色：主进程用本 job 的第0张“可见GPU”，prox 用本 job 的第1张
MAIN_LOCAL=0
PROX_LOCAL=1

LIP=5
ITER=1
REG=0.0001
MAXFUN=20
RECONOPT=2
STRIDE=1
START=0.98
PROXMODE=1

############################################################
# 2. 路径
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="/home/bozhen2/transcranial_pact_devito_model/V2_tesla"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 3. 初始化 module（Condor + Lmod 终极安全写法）
############################################################
set +u
export MODULEPATH=""
source /etc/profile.d/modules.sh
set -u

module purge
module use /etc/modulefiles/software
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

############################################################
# 4. Conda
############################################################
source /home/bozhen2/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

############################################################
# 5. GPU / Compiler  (SAFE on single node: auto-pick 2 free GPUs with locks)
############################################################
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 如果调度器注入了 CUDA_VISIBLE_DEVICES（有些集群会），就尊重它
echo "CUDA_VISIBLE_DEVICES (from scheduler) = ${CUDA_VISIBLE_DEVICES:-<unset>}"

# ---- GPU lock directory (node-local) ----
LOCK_DIR="/tmp/bozhen_gpu_locks"
mkdir -p "${LOCK_DIR}"

# 用 flock 锁住 GPU，避免你自己同时提交两个 job 抢同一张卡
# 注意：这是“同一台机器 c06 上”的锁，多个 job 都能看到 /tmp，所以有效。
_acquire_two_gpu_locks() {
  local max_try=50
  local try=0

  # 清空旧的 FD 变量
  unset GPU0_FD GPU1_FD MAIN_PHYS PROX_PHYS

  while [ $try -lt $max_try ]; do
    try=$((try+1))

    # 选 “显存占用最少”的 GPU（你们 c06 8 张 A40）
    # 输出类似： "0,33500" 表示 gpu0 已用 33.5GB（单位MiB）
    mapfile -t CANDS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sed 's/ //g' | sort -t, -k2,2n)

    # 尝试按从空闲到占用高的顺序抢两张锁
    local got=0
    local ids=()
    local fds=()

    for row in "${CANDS[@]}"; do
      local gid="${row%%,*}"
      local lockfile="${LOCK_DIR}/gpu${gid}.lock"

      # 尝试无阻塞加锁：成功则占用这张 GPU
      # 通过动态 FD（200+gid）保持锁一直持有到脚本退出
      local fd=$((200 + gid))
      eval "exec ${fd}>\"${lockfile}\""
      if flock -n "${fd}"; then
        ids+=("${gid}")
        fds+=("${fd}")
        got=$((got+1))
        if [ $got -eq 2 ]; then
          break
        fi
      else
        # 没抢到，关掉 fd
        eval "exec ${fd}>&-"
      fi
    done

    if [ $got -eq 2 ]; then
      MAIN_PHYS="${ids[0]}"
      PROX_PHYS="${ids[1]}"
      GPU0_FD="${fds[0]}"
      GPU1_FD="${fds[1]}"
      echo "Picked GPUs on $(hostname): MAIN_PHYS=${MAIN_PHYS}, PROX_PHYS=${PROX_PHYS}"
      return 0
    fi

    # 没抢到两张：释放已抢到的
    for fd in "${fds[@]}"; do
      eval "exec ${fd}>&-"
    done

    # 短暂重试（最多 50 次）
    sleep 1
  done

  echo "ERROR: cannot acquire 2 free GPU locks on $(hostname) after ${max_try} tries."
  return 1
}

# 清理：脚本退出时释放锁
_cleanup_gpu_locks() {
  if [ -n "${GPU0_FD:-}" ]; then eval "exec ${GPU0_FD}>&-"; fi
  if [ -n "${GPU1_FD:-}" ]; then eval "exec ${GPU1_FD}>&-"; fi
}
trap _cleanup_gpu_locks EXIT

# ---- Decide MAIN/PROX GPUs ----
# 如果 scheduler 没给 CUDA_VISIBLE_DEVICES，那就自己在 c06 上挑两张空闲的 GPU 并加锁
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _acquire_two_gpu_locks
else
  # scheduler 给了（比如 "2,3"），也做锁，防止你自己多 job 时覆盖
  IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  if [ "${#GPU_LIST[@]}" -lt 2 ]; then
    echo "ERROR: Need 2 GPUs but CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
    exit 1
  fi
  MAIN_PHYS="${GPU_LIST[0]}"
  PROX_PHYS="${GPU_LIST[1]}"
  # 给这两张也上锁（同机多 job 不会撞）
  for gid in "${MAIN_PHYS}" "${PROX_PHYS}"; do
    lockfile="${LOCK_DIR}/gpu${gid}.lock"
    fd=$((200 + gid))
    eval "exec ${fd}>\"${lockfile}\""
    if ! flock -n "${fd}"; then
      echo "ERROR: GPU ${gid} is already locked by another job on this node."
      exit 1
    fi
  done
fi

# 主进程只暴露 MAIN_PHYS
export CUDA_VISIBLE_DEVICES="${MAIN_PHYS}"
# prox worker 用 PROX_PHYS
export PROX_GPU_ID="${PROX_PHYS}"

echo "CUDA_VISIBLE_DEVICES(main)=${CUDA_VISIBLE_DEVICES}  PROX_GPU_ID=${PROX_GPU_ID}"

export CC=nvc
export CXX=nvc++


############################################################
# 6. Python path
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"

############################################################
# 7. SANITY CHECK
############################################################
echo "================ SANITY CHECK ================"
echo "HOST        : $(hostname)"
echo "PATH        : $PATH"
echo "CUDA_VISIBLE_DEVICES (main) : ${CUDA_VISIBLE_DEVICES}"
echo "PROX_GPU_ID (phys)          : ${PROX_GPU_ID}"

which dirname
which readlink
which python
which nvc++

nvc++ --version
nvidia-smi || true

python - <<'PY'
import os, sys
print("Python =", sys.executable)
try:
    import devito
    print("Devito =", devito.__version__)
except Exception as e:
    print("Devito import failed:", e)

try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    free, total = cp.cuda.runtime.memGetInfo()
    print("CuPy device_count =", n)
    print("CuPy device_id    =", dev.id)
    print("GPU name          =", props["name"].decode())
    print("mem free/total GB =", free/1e9, total/1e9)
except Exception as e:
    print("CuPy check failed:", e)

print("ENV CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("ENV PROX_GPU_ID          =", os.environ.get("PROX_GPU_ID"))
PY
echo "============================================="

############################################################
# 8. 日志
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxf${MAXFUN}_start${START}_str${STRIDE}_proxMode${PROXMODE}_recon${RECONOPT}_${TS}_CL.log"

############################################################
# 9. RUN（主进程：本 job 的 MAIN_PHYS）
############################################################
python -u "${PYTHON_SCRIPT}" \
  -i 3 \
  --skullp0 0 \
  --pressure nhp_3_nsp \
  -n 5 \
  -r ${REG} \
  -l ${LIP} \
  --recon_opt ${RECONOPT} \
  --iter ${ITER} \
  --maxfun ${MAXFUN} \
  --stride ${STRIDE} \
  --start ${START} \
  --prox_mode "${PROXMODE}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee "${LOG_FILE}"
