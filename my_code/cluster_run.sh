#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-devito}"

NEW_V2_ROOT="${NEW_V2_ROOT:-$HOME/Project/fix_versions/new_v2}"
REPO_ROOT="${REPO_ROOT:-$HOME/Project/fix_versions}"
PYTHON_MOD="${PYTHON_MOD:-new_v2.my_code.main}"
CONFIG_PATH="${CONFIG_PATH:-$NEW_V2_ROOT/my_code/cluster_config.json}"
WORKDIR="${WORKDIR:-$PWD}"
LOG_ROOT="${LOG_ROOT:-$WORKDIR/logs}"

mkdir -p "${LOG_ROOT}"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_ROOT}/${TS}.log"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CC=nvc
export CXX=nvc++
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

ASSIGNED_UUIDS="${NVIDIA_VISIBLE_DEVICES:-${_CONDOR_AssignedGPUs:-}}"
if [[ -z "${ASSIGNED_UUIDS}" ]]; then
  echo "ERROR: Condor did not provide NVIDIA_VISIBLE_DEVICES/_CONDOR_AssignedGPUs"
  env | grep -E 'NVIDIA_VISIBLE_DEVICES|_CONDOR_AssignedGPUs|CUDA_VISIBLE_DEVICES' || true
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
  exit 1
fi

IFS=',' read -ra UUIDS <<< "${ASSIGNED_UUIDS}"
if [[ "${#UUIDS[@]}" -lt 2 ]]; then
  echo "ERROR: Need 2 GPUs but got '${ASSIGNED_UUIDS}'"
  exit 1
fi

MAIN_UUID="${UUIDS[0]}"
PROX_UUID="${UUIDS[1]}"

export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"
export NVIDIA_VISIBLE_DEVICES="${MAIN_UUID}"
export PROX_CUDA_VISIBLE_DEVICES="${PROX_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_UUID}"

export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0

CPUSET="${CPUSET:-}"
TASKSET_PREFIX=()
if [[ -n "${CPUSET}" ]]; then
  TASKSET_PREFIX=(taskset -c "${CPUSET}")
fi

echo "================ RUN CONFIG ================"
echo "DATE        : $(date)"
echo "HOST        : $(hostname)"
echo "PWD         : $(pwd)"
echo "HOME        : ${HOME}"
echo "WORKDIR     : ${WORKDIR}"
echo "REPO_ROOT   : ${REPO_ROOT}"
echo "NEW_V2_ROOT : ${NEW_V2_ROOT}"
echo "PYTHON_MOD  : ${PYTHON_MOD}"
echo "PYTHONPATH  : ${PYTHONPATH}"
echo "CONFIG_PATH : ${CONFIG_PATH}"
echo "LOG_ROOT    : ${LOG_ROOT}"
echo "LOG_FILE    : ${LOG_FILE}"
echo "NVIDIA_VISIBLE_DEVICES : ${ASSIGNED_UUIDS}"
echo "MAIN_UUID   : ${MAIN_UUID}"
echo "PROX_UUID   : ${PROX_UUID}"
echo "============================================"
echo

echo "LOG_FILE=${LOG_FILE}"
echo "CONFIG=${CONFIG_PATH}"

"${TASKSET_PREFIX[@]}" python3 -u -m "${PYTHON_MOD}" --config "${CONFIG_PATH}"
