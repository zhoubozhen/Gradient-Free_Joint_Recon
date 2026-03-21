#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_ROOT="/home/bozhen2/my_packages/fista_tranPACT"

LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

DATE="$(date +"%Y%m%d_%H%M")"
LOG_FILE="${LOG_DIR}/${DATE}.log"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/exp_config.json}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}:${PYTHONPATH:-}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================ EXP LOCAL RUN CONFIG ================"
echo "DATE        : $(date)"
echo "HOST        : $(hostname)"
echo "SCRIPT_DIR  : ${SCRIPT_DIR}"
echo "WORKDIR     : ${WORKDIR}"
echo "PACKAGE_ROOT: ${PACKAGE_ROOT}"
echo "PYTHONPATH  : ${PYTHONPATH}"
echo "CONFIG_PATH : ${CONFIG_PATH}"
echo "LOG_FILE    : ${LOG_FILE}"
echo "======================================================"

echo
echo "================ CONFIG JSON BEGIN ================"
if [[ -f "${CONFIG_PATH}" ]]; then
  cat "${CONFIG_PATH}"
else
  echo "[WARN] CONFIG file not found: ${CONFIG_PATH}"
fi
echo "================= CONFIG JSON END ================="
echo

cd "${PACKAGE_ROOT}"
python3 -u my_code/exp_main.py --config "${CONFIG_PATH}" "$@"