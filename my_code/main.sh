#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="devito"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../new_v2/my_code
NEW_V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                 # .../new_v2
REPO_ROOT="$(cd "${NEW_V2_ROOT}/.." && pwd)"                  # repo root
PYTHON_MOD="new_v2.my_code.main"

# config 路径（按你当前：my_code/config.json）
CONFIG_PATH="${SCRIPT_DIR}/config.json"

# ---- module / conda ----
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export CC=nvc
export CXX=nvc++

# 工程标准：repo_root 在 PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config file not found: ${CONFIG_PATH}"
  exit 2
fi

# ---- read gpu indices from config.json (no external args) ----
MAIN_GPU_IDX="$(python - <<PY
import json
cfg=json.load(open(r"${CONFIG_PATH}"))
b=cfg.get("binding",{})
v=b.get("main_gpu_idx", None)
print("" if v is None else v)
PY
)"
PROX_GPU_IDX="$(python - <<PY
import json
cfg=json.load(open(r"${CONFIG_PATH}"))
b=cfg.get("binding",{})
v=b.get("prox_gpu_idx", None)
print("" if v is None else v)
PY
)"

if [[ -z "${MAIN_GPU_IDX}" || -z "${PROX_GPU_IDX}" ]]; then
  echo "ERROR: config missing binding.main_gpu_idx / binding.prox_gpu_idx"
  echo "CONFIG_PATH=${CONFIG_PATH}"
  exit 2
fi

# ---- auto cpuset based on gpu groups ----
CPUSET=""
if [[ "${MAIN_GPU_IDX}" -ge 0 && "${MAIN_GPU_IDX}" -le 2 && "${PROX_GPU_IDX}" -ge 0 && "${PROX_GPU_IDX}" -le 2 ]]; then
  CPUSET="0-15,32-47"     # a9 GPU0-2
elif [[ "${MAIN_GPU_IDX}" -ge 3 && "${MAIN_GPU_IDX}" -le 7 && "${PROX_GPU_IDX}" -ge 3 && "${PROX_GPU_IDX}" -le 7 ]]; then
  CPUSET="16-31,48-63"    # a9 GPU3-7
else
  echo "WARN: main_gpu_idx=${MAIN_GPU_IDX}, prox_gpu_idx=${PROX_GPU_IDX} not in same group."
  echo "WARN: Must both be in 0-2 or both be in 3-7 for NUMA binding. Will run WITHOUT taskset."
  CPUSET=""
fi

# ---- resolve physical index -> UUID ----
MAIN_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F',' -v g="${MAIN_GPU_IDX}" '$1+0==g {gsub(/ /,"",$2); print $2}')"
PROX_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F',' -v g="${PROX_GPU_IDX}" '$1+0==g {gsub(/ /,"",$2); print $2}')"

if [[ -z "${MAIN_UUID}" || -z "${PROX_UUID}" ]]; then
  echo "ERROR: Failed to resolve MAIN/PROX UUID from indices: main=${MAIN_GPU_IDX} prox=${PROX_GPU_IDX}"
  nvidia-smi -L || true
  exit 1
fi

echo "[GPU] MAIN_GPU_IDX=${MAIN_GPU_IDX} MAIN_UUID=${MAIN_UUID}"
echo "[GPU] PROX_GPU_IDX=${PROX_GPU_IDX} PROX_UUID=${PROX_UUID}"
echo "[CPU] CPUSET=${CPUSET:-<unset>}"

# main 只看到 1 张卡
export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"

# prox 子进程用（fista_tv_overall 会传给 prox worker）
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

# ---- write header + full config into log ----
{
  echo "================ RUN CONFIG ================="
  echo "DATE        : $(date)"
  echo "HOST        : $(hostname)"
  echo "PWD         : $(pwd)"
  echo "REPO_ROOT   : ${REPO_ROOT}"
  echo "NEW_V2_ROOT : ${NEW_V2_ROOT}"
  echo "PYTHONPATH  : ${PYTHONPATH}"
  echo "CONFIG_PATH : ${CONFIG_PATH}"
  echo "MAIN_GPU_IDX: ${MAIN_GPU_IDX}"
  echo "PROX_GPU_IDX: ${PROX_GPU_IDX}"
  echo "MAIN_UUID   : ${MAIN_UUID}"
  echo "PROX_UUID   : ${PROX_UUID}"
  echo "CUDA_VISIBLE_DEVICES(main) : ${CUDA_VISIBLE_DEVICES}"
  echo "PROX_CUDA_VISIBLE_DEVICES  : ${PROX_CUDA_VISIBLE_DEVICES}"
  echo "CPUSET      : ${CPUSET:-<unset>}"
  echo "--------------------------------------------"
  cat "${CONFIG_PATH}"
  echo "============================================"
  echo ""
} > "${LOG_FILE}"

echo "CONFIG=${CONFIG_PATH}"
echo "LOG_FILE=${LOG_FILE}"

# ---- run (append to LOG_FILE) ----
RUN_PREFIX=()
if [[ -n "${CPUSET}" ]]; then
  RUN_PREFIX=(taskset -c "${CPUSET}")
fi

"${RUN_PREFIX[@]}" \
python -u -m "${PYTHON_MOD}" --config "${CONFIG_PATH}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee -a "${LOG_FILE}"