#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${WORKDIR}/my_code/mpi_config.json}"

LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${DATE_TAG}.log"

# 先从 json 读取 saving_dir，这样同一份完整日志也能保存到 saving_dir/logs
SAVING_DIR="$(python3 - <<'PY' "${CONFIG_PATH}"
import json, sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

paths = cfg.get("paths", {}) or {}
saving_dir = paths.get("saving_dir", "")
if not saving_dir:
    raise SystemExit("[ERR] paths.saving_dir missing")

print(saving_dir)
PY
)"

SAVE_LOG_DIR="${SAVING_DIR}/logs"
mkdir -p "${SAVE_LOG_DIR}"
SAVE_LOG_FILE="${SAVE_LOG_DIR}/${DATE_TAG}.log"

touch "${LOG_FILE}" "${SAVE_LOG_FILE}"

# 从这里开始，所有 stdout/stderr 同时写两份完整日志
if [[ "${LOG_FILE}" == "${SAVE_LOG_FILE}" ]]; then
  exec > >(tee -a "${LOG_FILE}") 2>&1
else
  exec > >(tee -a "${LOG_FILE}" "${SAVE_LOG_FILE}") 2>&1
fi

module purge
module load nvidia-hpc-sdk-multi/25.1-rh8

source /home/bozhen2/anaconda3/etc/profile.d/conda.sh
conda activate devito

unset PYTHONHOME

export OMPI_ROOT="/software/nvidia-rh8-hpc-sdk-multi-25.1/Linux_x86_64/25.1/comm_libs/openmpi4"
export OPAL_PREFIX="${OMPI_ROOT}"
export PATH="${OMPI_ROOT}/bin:${PATH:-}"
export LD_LIBRARY_PATH="/software/gcc-5.3.0/lib64:${LD_LIBRARY_PATH:-}"

hash -r

export OMPI_CC=nvc
export OMPI_CXX=nvc++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}/src:${WORKDIR}:${PYTHONPATH:-}"

export DEVITO_LOGGING=ERROR
export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

readarray -t CFG_INFO < <(python3 - <<'PY' "${CONFIG_PATH}"
import json, sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

mpi = bool(cfg.get("mpi", False))
binding = cfg.get("binding", {}) or {}

main_gpu_idxs = binding.get("main_gpu_idxs", None)
main_gpu_idx  = binding.get("main_gpu_idx", None)
prox_gpu_idx  = binding.get("prox_gpu_idx", None)

if mpi:
    if main_gpu_idxs is None:
        if main_gpu_idx is not None:
            main_gpu_idxs = [main_gpu_idx]
        else:
            main_gpu_idxs = [0, 1]

    if not isinstance(main_gpu_idxs, list) or len(main_gpu_idxs) < 1:
        raise SystemExit("[ERR] binding.main_gpu_idxs invalid")

    np_rank = len(main_gpu_idxs)
    print("true")
    print(",".join(str(x) for x in main_gpu_idxs))
    print(str(np_rank))
else:
    if main_gpu_idx is None:
        if isinstance(main_gpu_idxs, list) and len(main_gpu_idxs) > 0:
            main_gpu_idx = main_gpu_idxs[0]
        else:
            main_gpu_idx = 0

    print("false")
    print(str(main_gpu_idx))
    print("1")

if prox_gpu_idx is None:
    prox_gpu_idx = 2

print(str(prox_gpu_idx))
PY
)

USE_MPI="${CFG_INFO[0]}"
MAIN_GPU_SPEC="${CFG_INFO[1]}"
NP="${NP:-${CFG_INFO[2]}}"
PROX_GPU_IDX="${CFG_INFO[3]}"

export WORKDIR
export CONFIG_PATH
export PROX_CUDA_VISIBLE_DEVICES="${PROX_GPU_IDX}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_GPU_IDX}"

{
  echo "================ RUN INFO ================"
  date
  echo "HOSTNAME=$(hostname)"
  echo "WORKDIR=${WORKDIR}"
  echo "CONFIG_PATH=${CONFIG_PATH}"
  echo "SAVING_DIR=${SAVING_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "SAVE_LOG_FILE=${SAVE_LOG_FILE}"
  echo "USE_MPI=${USE_MPI}"
  echo "MAIN_GPU_SPEC=${MAIN_GPU_SPEC}"
  echo "NP=${NP}"
  echo "PROX_CUDA_VISIBLE_DEVICES=${PROX_CUDA_VISIBLE_DEVICES}"
  echo "PROX_NVIDIA_VISIBLE_DEVICES=${PROX_NVIDIA_VISIBLE_DEVICES}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "PATH=${PATH}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "OPAL_PREFIX=${OPAL_PREFIX:-}"
  echo "which python3 = $(which python3)"
  echo "which mpirun  = $(which mpirun)"
  echo "which mpicc   = $(which mpicc)"
  echo "mpirun --version ="
  mpirun --version | head -n 5
  echo "mpicc --show ="
  mpicc --show
  echo
  echo "================ CONFIG BEGIN ================"
  cat "${CONFIG_PATH}"
  echo
  echo "================ CONFIG END =================="
  echo
}

cd "${WORKDIR}"

if [[ "${USE_MPI}" == "true" ]]; then
  export MAIN_GPU_LIST="${MAIN_GPU_SPEC}"

  mpirun -np "${NP}" \
    --map-by slot \
    --bind-to none \
    -x WORKDIR \
    -x CONFIG_PATH \
    -x MAIN_GPU_LIST \
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
    -x PROX_CUDA_VISIBLE_DEVICES \
    -x PROX_NVIDIA_VISIBLE_DEVICES \
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
      echo "[rank ${OMPI_COMM_WORLD_RANK}] CONFIG_PATH=${CONFIG_PATH}"
      echo "[rank ${OMPI_COMM_WORLD_RANK}] PROX_CUDA_VISIBLE_DEVICES=${PROX_CUDA_VISIBLE_DEVICES}"

      python3 -u "${WORKDIR}/my_code/main.py" --config "${CONFIG_PATH}"
    '
else
  export CUDA_VISIBLE_DEVICES="${MAIN_GPU_SPEC}"
  export NVIDIA_VISIBLE_DEVICES="${MAIN_GPU_SPEC}"

  export NV_ACC_DEVICE_TYPE=NVIDIA
  export NV_ACC_DEVICE_NUM=0
  export ACC_DEVICE_NUM=0

  echo "[single] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[single] CONFIG_PATH=${CONFIG_PATH}"
  echo "[single] PROX_CUDA_VISIBLE_DEVICES=${PROX_CUDA_VISIBLE_DEVICES}"

  python3 -u "${WORKDIR}/my_code/main.py" --config "${CONFIG_PATH}"
fi