#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-devito}"

# ====== 固定：日志永远写到 HOME（不会碰 /srv/condor/execute）======
LOG_ROOT="${LOG_ROOT:-$HOME/transcranial_pact_devito_model/new_v2/logs}"
mkdir -p "${LOG_ROOT}"
TS="$(date +"%Y%m%d_%H%M")"
LOG_FILE="${LOG_ROOT}/cluster_${TS}.log"

# ====== 工程路径：强制用 home 绝对路径，不用从脚本位置推导 ======
NEW_V2_ROOT="${NEW_V2_ROOT:-$HOME/transcranial_pact_devito_model/new_v2}"
REPO_ROOT="${REPO_ROOT:-$HOME/transcranial_pact_devito_model}"
PYTHON_MOD="${PYTHON_MOD:-new_v2.my_code.main}"

CONFIG_PATH="${CONFIG_PATH:-$NEW_V2_ROOT/my_code/config.json}"

# ---- module / conda ----
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CC=nvc
export CXX=nvc++

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# ====== GPU binding (Condor UUIDs) ======
ASSIGNED_UUIDS="${NVIDIA_VISIBLE_DEVICES:-${_CONDOR_AssignedGPUs:-}}"
if [[ -z "${ASSIGNED_UUIDS}" ]]; then
  echo "ERROR: Condor did not provide NVIDIA_VISIBLE_DEVICES/_CONDOR_AssignedGPUs" | tee -a "${LOG_FILE}"
  env | grep -E 'NVIDIA_VISIBLE_DEVICES|_CONDOR_AssignedGPUs|CUDA_VISIBLE_DEVICES' | tee -a "${LOG_FILE}" || true
  nvidia-smi -L | tee -a "${LOG_FILE}" || true
  exit 1
fi

IFS=',' read -ra UUIDS <<< "${ASSIGNED_UUIDS}"
if [[ "${#UUIDS[@]}" -lt 2 ]]; then
  echo "ERROR: Need 2 GPUs but got '${ASSIGNED_UUIDS}'" | tee -a "${LOG_FILE}"
  exit 1
fi

MAIN_UUID="${UUIDS[0]}"
PROX_UUID="${UUIDS[1]}"

export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"
export PROX_CUDA_VISIBLE_DEVICES="${PROX_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_UUID}"

# OpenACC（main 只看到 1 张卡 -> device 0）
export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0

# 可选：绑核（如果你需要）
CPUSET="${CPUSET:-}"
TASKSET_PREFIX=()
if [[ -n "${CPUSET}" ]]; then
  TASKSET_PREFIX=(taskset -c "${CPUSET}")
fi

# ---- header ----
{
  echo "================ RUN CONFIG ================"
  echo "DATE        : $(date)"
  echo "HOST        : $(hostname)"
  echo "PWD         : $(pwd)"
  echo "HOME        : ${HOME}"
  echo "REPO_ROOT   : ${REPO_ROOT}"
  echo "NEW_V2_ROOT : ${NEW_V2_ROOT}"
  echo "PYTHONPATH  : ${PYTHONPATH}"
  echo "CONFIG_PATH : ${CONFIG_PATH}"
  echo "NVIDIA_VISIBLE_DEVICES : ${ASSIGNED_UUIDS}"
  echo "MAIN_UUID   : ${MAIN_UUID}"
  echo "PROX_UUID   : ${PROX_UUID}"
  echo "CPUSET      : ${CPUSET:-<unset>}"
  echo "--------------------------------------------"
  if [[ -f "${CONFIG_PATH}" ]]; then
    cat "${CONFIG_PATH}"
  else
    echo "ERROR: config file not found: ${CONFIG_PATH}"
  fi
  echo "============================================"
  echo ""
} > "${LOG_FILE}"

echo "LOG_FILE=${LOG_FILE}"
echo "CONFIG=${CONFIG_PATH}"

# ---- run ----
"${TASKSET_PREFIX[@]}" \
python -u -m "${PYTHON_MOD}" --config "${CONFIG_PATH}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee -a "${LOG_FILE}"