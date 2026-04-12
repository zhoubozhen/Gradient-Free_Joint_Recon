#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-devito}"

NEW_V2_ROOT="${NEW_V2_ROOT:-/home/bozhen2/my_packages/fista_tranPACT}"
REPO_ROOT="${REPO_ROOT:-/home/bozhen2/my_packages}"
CONFIG_PATH="${CONFIG_PATH:-/home/bozhen2/Project/test/my_code/cluster_config.json}"
WORKDIR="${WORKDIR:-/home/bozhen2/Project/test}"
LOG_ROOT="${LOG_ROOT:-/home/bozhen2/Project/test/logs}"

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

unset PYTHONHOME

export OMPI_ROOT="/software/nvidia-rh8-hpc-sdk-multi-25.1/Linux_x86_64/25.1/comm_libs/openmpi4"
export OPAL_PREFIX="${OMPI_ROOT}"
export PATH="${OMPI_ROOT}/bin:${PATH:-}"
export LD_LIBRARY_PATH="/software/gcc-5.3.0/lib64:${LD_LIBRARY_PATH:-}"

hash -r

export CC=nvc
export CXX=nvc++
export OMPI_CC=nvc
export OMPI_CXX=nvc++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

# 你既然 main.py 里固定插 mpi_fista_tranPACT/src，
# 这里不要再把别的 src 放到最前面，避免混两套代码
export PYTHONPATH="${NEW_V2_ROOT}/src:${NEW_V2_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

export DEVITO_LOGGING=ERROR
export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

ASSIGNED_UUIDS="${NVIDIA_VISIBLE_DEVICES:-${_CONDOR_AssignedGPUs:-}}"
if [[ -z "${ASSIGNED_UUIDS}" ]]; then
  echo "ERROR: Condor did not provide NVIDIA_VISIBLE_DEVICES/_CONDOR_AssignedGPUs"
  env | grep -E 'NVIDIA_VISIBLE_DEVICES|_CONDOR_AssignedGPUs|CUDA_VISIBLE_DEVICES' || true
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
  exit 1
fi

IFS=',' read -ra UUIDS <<< "${ASSIGNED_UUIDS}"
NGPU="${#UUIDS[@]}"

if [[ "${NGPU}" -lt 3 ]]; then
  echo "ERROR: Need 3 GPUs (2 main + 1 prox), got '${ASSIGNED_UUIDS}'"
  exit 1
fi

NP="${NP:-2}"

MAIN_GPU_LIST="${UUIDS[0]},${UUIDS[1]}"
PROX_GPU_UUID="${UUIDS[2]}"

export WORKDIR
export CONFIG_PATH
export MAIN_GPU_LIST
export PROX_CUDA_VISIBLE_DEVICES="${PROX_GPU_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_GPU_UUID}"

{
  echo "================ RUN CONFIG ================"
  echo "DATE        : $(date)"
  echo "HOST        : $(hostname)"
  echo "PWD         : $(pwd)"
  echo "HOME        : ${HOME}"
  echo "WORKDIR     : ${WORKDIR}"
  echo "REPO_ROOT   : ${REPO_ROOT}"
  echo "NEW_V2_ROOT : ${NEW_V2_ROOT}"
  echo "PYTHONPATH  : ${PYTHONPATH}"
  echo "CONFIG_PATH : ${CONFIG_PATH}"
  echo "LOG_ROOT    : ${LOG_ROOT}"
  echo "LOG_FILE    : ${LOG_FILE}"
  echo "ASSIGNED_UUIDS : ${ASSIGNED_UUIDS}"
  echo "MAIN_GPU_LIST  : ${MAIN_GPU_LIST}"
  echo "PROX_GPU_UUID  : ${PROX_GPU_UUID}"
  echo "NP             : ${NP}"
  echo "which python3  : $(which python3)"
  echo "which mpirun   : $(which mpirun)"
  echo "which mpicc    : $(which mpicc)"
  echo "mpirun --version:"
  mpirun --version | head -n 5
  echo "mpicc --show:"
  mpicc --show
  echo "============================================"
  echo
  echo "================ CONFIG BEGIN ================"
  cat "${CONFIG_PATH}"
  echo "================= CONFIG END ================="
  echo
}

cd "${WORKDIR}"

mpirun -np "${NP}" \
  --map-by slot \
  --bind-to none \
  -x WORKDIR \
  -x CONFIG_PATH \
  -x MAIN_GPU_LIST \
  -x PROX_CUDA_VISIBLE_DEVICES \
  -x PROX_NVIDIA_VISIBLE_DEVICES \
  -x PYTHONPATH \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x OPAL_PREFIX \
  -x CUDA_DEVICE_ORDER \
  -x PYTHONUNBUFFERED \
  -x DEVITO_LOGGING \
  -x DEVITO_LANGUAGE \
  -x DEVITO_ARCH \
  -x DEVITO_PLATFORM \
  -x OMPI_CC \
  -x OMPI_CXX \
  bash -lc '
    set -euo pipefail
    IFS=, read -r -a GPUS <<< "$MAIN_GPU_LIST"

    if [[ ${OMPI_COMM_WORLD_RANK} -ge ${#GPUS[@]} ]]; then
      echo "[ERR] rank ${OMPI_COMM_WORLD_RANK} out of range for MAIN_GPU_LIST=${MAIN_GPU_LIST}"
      exit 1
    fi

    export CUDA_VISIBLE_DEVICES="${GPUS[$OMPI_COMM_WORLD_RANK]}"
    export NVIDIA_VISIBLE_DEVICES="${GPUS[$OMPI_COMM_WORLD_RANK]}"

    export NV_ACC_DEVICE_TYPE=NVIDIA
    export NV_ACC_DEVICE_NUM=0
    export ACC_DEVICE_NUM=0

    echo "[rank ${OMPI_COMM_WORLD_RANK}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[rank ${OMPI_COMM_WORLD_RANK}] PROX_CUDA_VISIBLE_DEVICES=${PROX_CUDA_VISIBLE_DEVICES}"

    python3 -u "${WORKDIR}/my_code/main.py" --config "${CONFIG_PATH}"
  '