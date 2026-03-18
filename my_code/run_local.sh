#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

DATE="$(date +%Y%m%d)"
LOG_FILE="${LOG_DIR}/${DATE}.log"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run_local] START $(date)"
echo "[run_local] WORKDIR=${WORKDIR}"
echo "[run_local] LOG_FILE=${LOG_FILE}"

cd "${WORKDIR}"
python3 -u my_code/main.py --config my_code/config.json "$@"
