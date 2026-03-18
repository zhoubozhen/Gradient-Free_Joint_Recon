#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

DATE="$(date +"%Y%m%d_%H%M")"
LOG_FILE="${LOG_DIR}/${DATE}.log"
CONFIG_PATH="${CONFIG_PATH:-my_code/config.json}"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run_local] START $(date)"
echo "[run_local] WORKDIR=${WORKDIR}"
echo "[run_local] LOG_FILE=${LOG_FILE}"

cd "${WORKDIR}"
echo ""
echo "CONFIG=${CONFIG_PATH}"
echo "================ CONFIG BEGIN ================"
if [[ -f "${CONFIG_PATH}" ]]; then
  cat "${CONFIG_PATH}"
else
  echo "[WARN] CONFIG file not found: ${CONFIG_PATH}"
fi
echo "================= CONFIG END ================="
echo ""

python3 -u my_code/main.py --config "${CONFIG_PATH}" "$@"
