#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="devito"

# ========== 你只需要改这几个（非 Condor 运行时有效）==========
MAIN_GPU_IDX="${MAIN_GPU_IDX:-6}"     # 物理 GPU index（非 Condor）
PROX_GPU_IDX="${PROX_GPU_IDX:-7}"     # 物理 GPU index（非 Condor）
CPUSET="${CPUSET:-16-31,48-63}"       # 共享机绑核（可选）
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../new_v2/my_code
NEW_V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                 # .../new_v2
REPO_ROOT="$(cd "${NEW_V2_ROOT}/.." && pwd)"                  # repo root
PYTHON_MOD="new_v2.my_code.main"

# 你现在把 config 放到了 my_code/config/ 目录，并且改名为 config（建议用 .json 结尾）
# 这里按你截图：my_code/config/config.json
CONFIG_PATH="${SCRIPT_DIR}/config.json"

# ---- module / conda ----
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export CC=nvc
export CXX=nvc++

# 工程标准：让 repo_root 在 PYTHONPATH，这样 `import new_v2...` 永远 OK
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ---- GPU binding (auto detect condor vs local) ----
# Condor 通常会给 NVIDIA_VISIBLE_DEVICES="GPU-uuid1,GPU-uuid2"
ASSIGNED_UUIDS="${NVIDIA_VISIBLE_DEVICES:-${_CONDOR_AssignedGPUs:-}}"

if [[ -n "${ASSIGNED_UUIDS}" ]]; then
  # Condor 路径：直接用分配的 UUID 列表
  IFS=',' read -ra UUIDS <<< "${ASSIGNED_UUIDS}"
  if [[ "${#UUIDS[@]}" -lt 2 ]]; then
    echo "ERROR: Need 2 GPUs but got NVIDIA_VISIBLE_DEVICES='${ASSIGNED_UUIDS}'"
    exit 1
  fi
  MAIN_UUID="${UUIDS[0]}"
  PROX_UUID="${UUIDS[1]}"
else
  # 非 Condor：用物理 index -> UUID
  MAIN_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F',' -v g="${MAIN_GPU_IDX}" '$1+0==g {gsub(/ /,"",$2); print $2}')"
  PROX_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F',' -v g="${PROX_GPU_IDX}" '$1+0==g {gsub(/ /,"",$2); print $2}')"
fi

if [[ -z "${MAIN_UUID}" || -z "${PROX_UUID}" ]]; then
  echo "ERROR: Failed to resolve MAIN/PROX UUID"
  nvidia-smi -L || true
  exit 1
fi

echo "[GPU] MAIN_UUID=${MAIN_UUID}"
echo "[GPU] PROX_UUID=${PROX_UUID}"

# main 只看到 1 张卡
export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"

# prox 子进程用（prox worker 内部读取并设置 CUDA_VISIBLE_DEVICES）
export PROX_CUDA_VISIBLE_DEVICES="${PROX_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_UUID}"

# OpenACC（main 只暴露 1 张卡 -> device 0）
export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0

# ---- logging ----
LOG_DIR="${NEW_V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +"%Y%m%d_%H%M")"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

# ---- 写入 config 头部（你要的功能）----
{
  echo "================ RUN CONFIG ================="
  echo "DATE        : $(date)"
  echo "HOST        : $(hostname)"
  echo "PWD         : $(pwd)"
  echo "REPO_ROOT   : ${REPO_ROOT}"
  echo "NEW_V2_ROOT : ${NEW_V2_ROOT}"
  echo "PYTHONPATH  : ${PYTHONPATH}"
  echo "CONFIG_PATH : ${CONFIG_PATH}"
  echo "MAIN_UUID   : ${MAIN_UUID}"
  echo "PROX_UUID   : ${PROX_UUID}"
  echo "CPUSET      : ${CPUSET}"
  echo "--------------------------------------------"
  if [[ -f "${CONFIG_PATH}" ]]; then
    cat "${CONFIG_PATH}"
  else
    echo "ERROR: config file not found: ${CONFIG_PATH}"
  fi
  echo "============================================"
  echo ""
} > "${LOG_FILE}"

echo "================ RUN ================"
echo "CONFIG=${CONFIG_PATH}"
echo "LOG_FILE=${LOG_FILE}"
echo "====================================="

# ---- run (append 到 LOG_FILE) ----
# 整个进程树绑核（main + prox 都继承）
taskset -c "${CPUSET}" \
python -u -m "${PYTHON_MOD}" --config "${CONFIG_PATH}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee -a "${LOG_FILE}"