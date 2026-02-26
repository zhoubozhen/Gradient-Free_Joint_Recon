#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="devito"

# ===== 本地 debug 固定参数（不从外部读任何参数）=====
GPU_IDX="0"
CPUSET=""          # 不绑核就留空；要绑核就写 "16-31,48-63"
DEBUG_PORT="5678"
WAIT_FOR_CLIENT="1"  # 1=等待VSCode attach；0=不等待直接跑

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../new_v2/my_code
NEW_V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                 # .../new_v2
REPO_ROOT="$(cd "${NEW_V2_ROOT}/.." && pwd)"                  # repo root
PYTHON_MOD="new_v2.my_code.main"

# 你的本地 config（按你当前放法：my_code/config.json）
CONFIG_PATH="${SCRIPT_DIR}/config.json"

# ---- module / conda ----
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export CC=nvc
export CXX=nvc++

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${GPU_IDX}"

# OpenACC（只暴露 1 张卡 -> device 0）
export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0

# ---- logging ----
LOG_DIR="${NEW_V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +"%Y%m%d_%H%M")"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

# ---- 写入 header + config ----
{
  echo "================ RUN CONFIG ================="
  echo "DATE        : $(date)"
  echo "HOST        : $(hostname)"
  echo "PWD         : $(pwd)"
  echo "REPO_ROOT   : ${REPO_ROOT}"
  echo "NEW_V2_ROOT : ${NEW_V2_ROOT}"
  echo "PYTHONPATH  : ${PYTHONPATH}"
  echo "CONFIG_PATH : ${CONFIG_PATH}"
  echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
  echo "CPUSET      : ${CPUSET:-<unset>}"
  echo "DEBUG_PORT  : ${DEBUG_PORT}"
  echo "WAIT_CLIENT : ${WAIT_FOR_CLIENT}"
  echo "--------------------------------------------"
  if [[ -f "${CONFIG_PATH}" ]]; then
    cat "${CONFIG_PATH}"
  else
    echo "ERROR: config file not found: ${CONFIG_PATH}"
    exit 2
  fi
  echo "============================================"
  echo ""
} > "${LOG_FILE}"

echo "CONFIG=${CONFIG_PATH}"
echo "LOG_FILE=${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---- run ----
RUN_PREFIX=()
if [[ -n "${CPUSET}" ]]; then
  RUN_PREFIX=(taskset -c "${CPUSET}")
fi

DBG_ARGS=( -m debugpy --listen "${DEBUG_PORT}" )
if [[ "${WAIT_FOR_CLIENT}" == "1" ]]; then
  DBG_ARGS+=( --wait-for-client )
fi

"${RUN_PREFIX[@]}" \
python -u "${DBG_ARGS[@]}" \
  -m "${PYTHON_MOD}" --config "${CONFIG_PATH}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee -a "${LOG_FILE}"