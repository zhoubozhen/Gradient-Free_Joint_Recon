#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUR_DIR="$PWD"
if [[ "$(basename "$CUR_DIR")" == "my_code" ]]; then
  TARGET_DIR="$(cd "$CUR_DIR/.." && pwd)"
else
  TARGET_DIR="$CUR_DIR"
fi

echo "Initializing EXP workdir at:"
echo "  ${TARGET_DIR}"
echo ""
echo "PACKAGE_ROOT=${PACKAGE_ROOT}"
echo ""

mkdir -p "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/logs"
mkdir -p "${TARGET_DIR}/output"
mkdir -p "${TARGET_DIR}/my_code"

copy_if_missing () {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" ]]; then
    echo "[skip] $dst already exists"
  else
    cp "$src" "$dst"
    echo "[copy] $dst"
  fi
}

copy_if_missing "${SCRIPT_DIR}/exp_main.py"     "${TARGET_DIR}/my_code/exp_main.py"
copy_if_missing "${SCRIPT_DIR}/exp_config.json" "${TARGET_DIR}/my_code/exp_config.json"

cat > "${TARGET_DIR}/my_code/run_exp_local.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="\$(cd "\${SCRIPT_DIR}/.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT}"

LOG_DIR="\${WORKDIR}/logs"
mkdir -p "\${LOG_DIR}"

DATE="\$(date +"%Y%m%d_%H%M")"
LOG_FILE="\${LOG_DIR}/\${DATE}.log"
CONFIG_PATH="\${CONFIG_PATH:-\${SCRIPT_DIR}/exp_config.json}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="\${WORKDIR}:\${PACKAGE_ROOT}/src:\${PACKAGE_ROOT}:\${PYTHONPATH:-}"

exec > >(tee -a "\${LOG_FILE}") 2>&1

echo "================ EXP LOCAL RUN CONFIG ================"
echo "DATE        : \$(date)"
echo "HOST        : \$(hostname)"
echo "SCRIPT_DIR  : \${SCRIPT_DIR}"
echo "WORKDIR     : \${WORKDIR}"
echo "PACKAGE_ROOT: \${PACKAGE_ROOT}"
echo "PYTHONPATH  : \${PYTHONPATH}"
echo "CONFIG_PATH : \${CONFIG_PATH}"
echo "LOG_FILE    : \${LOG_FILE}"
echo "======================================================"

echo
echo "================ CONFIG JSON BEGIN ================"
if [[ -f "\${CONFIG_PATH}" ]]; then
  cat "\${CONFIG_PATH}"
else
  echo "[WARN] CONFIG file not found: \${CONFIG_PATH}"
fi
echo "================= CONFIG JSON END ================="
echo

cd "\${WORKDIR}"
python3 -u my_code/exp_main.py --config "\${CONFIG_PATH}" "\$@"
EOF

chmod +x "${TARGET_DIR}/my_code/exp_main.py" "${TARGET_DIR}/my_code/run_exp_local.sh"

echo ""
echo "Done."
echo "Run next:"
echo "  cd ${TARGET_DIR}/my_code"
echo "  bash run_exp_local.sh"