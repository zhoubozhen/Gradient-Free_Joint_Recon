#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="devito"
############################################################
# 0. 基本参数（Red Hat 原环境）
############################################################
# GPU="4"
# ---------  参数（只影响 Python 逻辑）---------
LIP=5
ITER=20
REG=0.0001
MAXFUN=60
RECONOPT=2
STRIDE=1
START=1.05
PROXMODE=2 # 2启动prox子进程在PROX_GPU上跑
MAIN_GPU="3" 
PROX_GPU="4" # 如果PROXMODE=2，必须指定，如果是1，就没必要
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
# 5. GPU 绑定（本地 machine：main=GPU4, prox=GPU5）
############################################################
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# 查询 UUID（物理编号 -> UUID）
MAIN_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F',' -v g="${MAIN_GPU}" '$1+0==g {gsub(/ /,"",$2); print $2}')

PROX_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F',' -v g="${PROX_GPU}" '$1+0==g {gsub(/ /,"",$2); print $2}')

if [ -z "${MAIN_UUID}" ] || [ -z "${PROX_UUID}" ]; then
  echo "ERROR: Failed to resolve GPU UUIDs (MAIN=${MAIN_GPU}, PROX=${PROX_GPU})"
  nvidia-smi -L
  exit 1
fi

echo "[LOCAL GPU MAP] MAIN_GPU=${MAIN_GPU} -> ${MAIN_UUID}"
echo "[LOCAL GPU MAP] PROX_GPU=${PROX_GPU} -> ${PROX_UUID}"

# main 进程：只看见 GPU4
export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"

# prox subprocess：只看见 GPU5（fista_tv_overall.py 会用）
export PROX_CUDA_VISIBLE_DEVICES="${PROX_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_UUID}"

# OpenACC（main 只有 1 张可见卡 => index 0）
export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0


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
# 7. Run（参数改为 Tesla 风格）
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxf${MAXFUN}_start${START}_str${STRIDE}_recon${RECONOPT}_proxMode${PROXMODE}_${TS}_RH.log"

echo "================ RUN (Red Hat, NO MPI) ================"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SCRIPT=${PYTHON_SCRIPT}"
echo "LOG_FILE=${LOG_FILE}"
echo "PARAM: LIP=${LIP} ITER=${ITER} REG=${REG} MAXFUN=${MAXFUN}"
echo "======================================================"

python -u "${PYTHON_SCRIPT}" \
  -i 3 \
  --skullp0 0 \
  --pressure nhp_3_nsp \
  -n 5 \
  -r "${REG}" \
  -l "${LIP}" \
  --recon_opt "${RECONOPT}" \
  --iter "${ITER}" \
  --maxfun "${MAXFUN}"\
  --stride "${STRIDE}"\
  --start "${START}"\
  --prox_mode "${PROXMODE}"\
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee "${LOG_FILE}"