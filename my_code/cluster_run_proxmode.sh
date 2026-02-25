#!/usr/bin/env bash
set -euo pipefail
set -x
############################################################
# 0. Condor execute node PATH 修复（只补，不覆盖）
############################################################
if ! command -v dirname >/dev/null 2>&1; then
  export PATH="/usr/bin:/bin:/usr/local/bin:${PATH:-}"
fi

ENV_NAME=devito
############################################################
# 1. 参数
############################################################
LIP=5
ITER=1
REG=0.0001
MAXFUN=20
RECONOPT=2
STRIDE=1
START=0.98
PROXMODE=2 # prox_mode=1: 原逻辑（同进程/同GPU）
# prox_mode=2: 走 subprocess prox worker（双卡）
############################################################
# 2. 路径
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="/home/bozhen2/transcranial_pact_devito_model/V2_tesla"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 3. 初始化 module（Condor + Lmod 安全写法）
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
# 5. GPU / Condor 分配信息 & 物理卡映射打印
############################################################
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "================ CONDOR / ENV SNAPSHOT ================"
echo "HOST                     : $(hostname)"
echo "DATE                     : $(date)"
echo "USER                     : $(whoami)"
echo "PWD                      : $(pwd)"
echo "CONDOR_JOB_ID            : ${CONDOR_JOB_ID:-<unset>}"
echo "ClusterId.ProcId         : ${ClusterId:-<unset>}.${ProcId:-<unset>}"
echo "CUDA_DEVICE_ORDER        : ${CUDA_DEVICE_ORDER:-<unset>}"
echo "CUDA_VISIBLE_DEVICES(in) : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES   : ${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "_CONDOR_AssignedGPUs     : ${_CONDOR_AssignedGPUs:-<unset>}"
echo "--------------------------------------------------------"
echo "[ENV grep -E 'CUDA|NVIDIA|CONDOR|_CONDOR']"
env | grep -E '^(CUDA|NVIDIA|CONDOR|_CONDOR)' || true
echo "========================================================"

echo "================ GPU INVENTORY (PHYSICAL) ================"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total --format=csv,noheader || true
echo "=========================================================="

# ----------------------------------------------------------
# NEW: derive assigned UUIDs from NVIDIA_VISIBLE_DEVICES / _CONDOR_AssignedGPUs
# Condor+nvidia-container-runtime 通常会把这两个 UUID 写在 NVIDIA_VISIBLE_DEVICES
# ----------------------------------------------------------
ASSIGNED_UUIDS="${NVIDIA_VISIBLE_DEVICES:-${_CONDOR_AssignedGPUs:-}}"
if [ -z "${ASSIGNED_UUIDS}" ]; then
  echo "ERROR: NVIDIA_VISIBLE_DEVICES/_CONDOR_AssignedGPUs empty; cannot get assigned GPU UUIDs"
  exit 1
fi

# Normalize: comma-separated list
IFS=',' read -ra UUIDS <<< "${ASSIGNED_UUIDS}"

if [ "${#UUIDS[@]}" -lt 2 ]; then
  echo "ERROR: Need 2 GPUs but assigned UUID list='${ASSIGNED_UUIDS}'"
  exit 1
fi

MAIN_UUID="${UUIDS[0]}"
PROX_UUID="${UUIDS[1]}"

echo "================ GPU ALLOCATION (ASSIGNED UUIDS) =========="
echo "ASSIGNED_UUIDS = ${ASSIGNED_UUIDS}"
echo "MAIN_UUID      = ${MAIN_UUID}"
echo "PROX_UUID      = ${PROX_UUID}"
echo "==========================================================="

# ----------------------------------------------------------
# MAIN: only see MAIN_UUID (single visible GPU => local index is always 0)
# PROX worker: will be launched with only PROX_UUID visible => local index is always 0
# ----------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"

# These are read by fista_tv_overall.py for subprocess worker binding
export PROX_CUDA_VISIBLE_DEVICES="${PROX_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_UUID}"

# OpenACC bind (MAIN) - single visible GPU => device 0
export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0

echo "================ LOCAL GPU VIEW (MAIN) ==================="
echo "After UUID binding:"
echo "  CUDA_VISIBLE_DEVICES(main) = ${CUDA_VISIBLE_DEVICES}"
echo "  PROX_CUDA_VISIBLE_DEVICES  = ${PROX_CUDA_VISIBLE_DEVICES}"
echo "  PROX_NVIDIA_VISIBLE_DEVICES= ${PROX_NVIDIA_VISIBLE_DEVICES}"
echo "  NV_ACC_DEVICE_NUM(main)    = ${NV_ACC_DEVICE_NUM}"
echo "  ACC_DEVICE_NUM(main)       = ${ACC_DEVICE_NUM}"
echo "-----------------------------------------------------------"
echo "[GPU CHECK] MAIN local view (should be 1 visible GPU in CUDA runtime):"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "==========================================================="

export CC=nvc
export CXX=nvc++

############################################################
# 6. Python path
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"
export V2_ROOT="${V2_ROOT}"

############################################################
# 7. SANITY CHECK
############################################################
echo "================ SANITY CHECK ================"
which python || true
which nvc++ || true
nvc++ --version || true
echo "---- nvidia-smi (MAIN view) ----"
nvidia-smi || true
nvidia-smi pmon -c 1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader || true
echo "======================================="

############################################################
# 8. 日志
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxf${MAXFUN}_start${START}_str${STRIDE}_proxMode${PROXMODE}_recon${RECONOPT}_${TS}_CL.log"

############################################################
# 9. RUN
############################################################
echo "================ RUN MAIN ==================="
echo "LOG_FILE = ${LOG_FILE}"
echo "================================================"

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

echo "================ POST-RUN SNAPSHOT ======================="
nvidia-smi || true
nvidia-smi pmon -c 1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader || true
echo "=========================================================="
