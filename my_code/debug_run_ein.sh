#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="devito"
############################################################
# 0. 基本参数（Red Hat 原环境）
############################################################
GPU="0"
# ---------  参数（只影响 Python 逻辑）---------
LIP=5
ITER=20
REG=0.0001
MAXFUN=60
RECONOPT=2
STRIDE=1
START=1.05
# ------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 1. module 环境（⚠️ 完全保持原 Red Hat）
############################################################
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

############################################################
# 2. conda（原样）
############################################################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

############################################################
# 3. 编译器（原样）
############################################################
export CC=nvc
export CXX=nvc++

############################################################
# 4. PYTHONPATH（原样）
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"

############################################################
# 5. GPU 绑定（原样）
############################################################
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${GPU}"

############################################################
# 6. SANITY CHECK（原样）
############################################################
echo "================ SANITY CHECK ================"
which python
python - << 'EOF'
import importlib, sys
print("Python:", sys.executable)
print("sys.path (head):")
for p in sys.path[:8]:
    print(" ", p)

import tranPACT
print("tranPACT OK ->", tranPACT.__file__)

m = importlib.import_module("fista_tv_3d_python.fista_tv_overall")
print("fista_tv_3d_python.fista_tv_overall ->", m.__file__)

mods = [
    "fista_tv_3d_python.proximal_L_cupy_mix",
    "fista_tv_3d_python.proximal_L",
    "fista_tv_3d_python.proximal_L_cupy",
    "fista_tv_3d_python.proximal_L_2d",
    "fista_tv_3d_python.cost_func_tv",
]
for x in mods:
    mm = importlib.import_module(x)
    print("[OK]", x, "->", mm.__file__)

print("SANITY CHECK OK")
EOF
echo "=============================================="

############################################################
# 7. Run（DEBUG via VS Code）
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxf${MAXFUN}_start${START}_str${STRIDE}_recon${RECONOPT}_${TS}_ein.log"

echo "================ RUN (DEBUG, NO MPI) ================"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SCRIPT=${PYTHON_SCRIPT}"
echo "LOG_FILE=${LOG_FILE}"
echo "PARAM: LIP=${LIP} ITER=${ITER} REG=${REG} MAXFUN=${MAXFUN}"
echo "====================================================="

python -u -m debugpy \
  --listen 5678 \
  --wait-for-client \
  "${PYTHON_SCRIPT}" \
  -i 3 \
  --skullp0 0 \
  --pressure nhp_3_nsp \
  -n 5 \
  -r "${REG}" \
  -l "${LIP}" \
  --recon_opt "${RECONOPT}" \
  --iter "${ITER}" \
  --maxfun "${MAXFUN}" \
  --stride "${STRIDE}" \
  --start "${START}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee "${LOG_FILE}"
