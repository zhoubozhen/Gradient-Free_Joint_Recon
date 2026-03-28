#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${WORKDIR}/my_code/mpi_config.json}"
LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${DATE_TAG}.log"

source /home/bozhen2/anaconda3/etc/profile.d/conda.sh
conda activate devito

module unload openmpi/rhel/5.0.7 2>/dev/null || true
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2
module load openmpi/rhel/5.0.8-cuda-aware
hash -r

export OMPI_CC=nvc
export OMPI_CXX=nvc++
export MPICC=nvc
export MPICXX=nvc++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
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
  echo "USE_MPI=${USE_MPI}"
  echo "MAIN_GPU_SPEC=${MAIN_GPU_SPEC}"
  echo "NP=${NP}"
  echo "PROX_CUDA_VISIBLE_DEVICES=${PROX_CUDA_VISIBLE_DEVICES}"
  echo "PROX_NVIDIA_VISIBLE_DEVICES=${PROX_NVIDIA_VISIBLE_DEVICES}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo
  echo "================ CONFIG BEGIN ================"
  cat "${CONFIG_PATH}"
  echo
  echo "================ CONFIG END =================="
  echo
} | tee "${LOG_FILE}"

cd "${WORKDIR}"

if [[ "${USE_MPI}" == "true" ]]; then
  export MAIN_GPU_LIST="${MAIN_GPU_SPEC}"

  mpirun -np "${NP}" \
    --map-by slot \
    --bind-to none \
    --oversubscribe \
    -x WORKDIR \
    -x CONFIG_PATH \
    -x MAIN_GPU_LIST \
    -x PYTHONPATH \
    -x CUDA_DEVICE_ORDER \
    -x DEVITO_LOGGING \
    -x DEVITO_LANGUAGE \
    -x DEVITO_ARCH \
    -x DEVITO_PLATFORM \
    -x PROX_CUDA_VISIBLE_DEVICES \
    -x PROX_NVIDIA_VISIBLE_DEVICES \
    -x OMPI_CC \
    -x OMPI_CXX \
    -x MPICC \
    -x MPICXX \
    bash -lc '
      IFS=, read -r -a GPUS <<< "$MAIN_GPU_LIST"
      export CUDA_VISIBLE_DEVICES="${GPUS[$OMPI_COMM_WORLD_RANK]}"
      export NVIDIA_VISIBLE_DEVICES="${GPUS[$OMPI_COMM_WORLD_RANK]}"
      export NV_ACC_DEVICE_NUM=0
      export ACC_DEVICE_NUM=0
      echo "[rank ${OMPI_COMM_WORLD_RANK}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CONFIG_PATH=${CONFIG_PATH}"
      python3 -u "${WORKDIR}/my_code/main.py" --config "${CONFIG_PATH}"
    ' 2>&1 | tee -a "${LOG_FILE}"
else
  export CUDA_VISIBLE_DEVICES="${MAIN_GPU_SPEC}"
  export NVIDIA_VISIBLE_DEVICES="${MAIN_GPU_SPEC}"
  export NV_ACC_DEVICE_NUM=0
  export ACC_DEVICE_NUM=0

  echo "[single] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CONFIG_PATH=${CONFIG_PATH}" | tee -a "${LOG_FILE}"
  python3 -u "${WORKDIR}/my_code/main.py" --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_FILE}"
fi