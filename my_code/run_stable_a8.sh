#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="devito"
############################################################
# 0. 基本参数
############################################################
LIP=5
ITER=20
REG=0.0001
MAXFUN=60
RECONOPT=2
STRIDE=1
START=1.05
PROXMODE=2            # 2 = 启动 prox 子进程
MAIN_GPU="6"          # main 用的物理 GPU
PROX_GPU="7"          # prox 用的物理 GPU（仅 PROXMODE=2 需要）
CPUSET="16-31,48-63" # CPU 绑定（a9 GPU3-7 位于 NUMA node1）
# CPUSET="0-15,32-47" #（a9 GPU0-2 位于 NUMA node1）ein必选这个
############################################################
# 路径
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 1. module 环境
############################################################
module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

############################################################
# 2. conda
############################################################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

############################################################
# 3. 编译器
############################################################
export CC=nvc
export CXX=nvc++

############################################################
# 4. PYTHONPATH
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"

############################################################
# 5. GPU 绑定（UUID，main / prox 隔离）
############################################################
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

MAIN_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F',' -v g="${MAIN_GPU}" '$1+0==g {gsub(/ /,"",$2); print $2}')

PROX_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F',' -v g="${PROX_GPU}" '$1+0==g {gsub(/ /,"",$2); print $2}')

if [ -z "${MAIN_UUID}" ] || [ -z "${PROX_UUID}" ]; then
  echo "ERROR: Failed to resolve GPU UUIDs (MAIN=${MAIN_GPU}, PROX=${PROX_GPU})"
  nvidia-smi -L
  exit 1
fi

echo "[GPU MAP] MAIN_GPU=${MAIN_GPU} -> ${MAIN_UUID}"
echo "[GPU MAP] PROX_GPU=${PROX_GPU} -> ${PROX_UUID}"

# main 进程
export CUDA_VISIBLE_DEVICES="${MAIN_UUID}"

# prox 子进程（由 Python 继承）
export PROX_CUDA_VISIBLE_DEVICES="${PROX_UUID}"
export PROX_NVIDIA_VISIBLE_DEVICES="${PROX_UUID}"

# OpenACC（main 只有 1 张卡）
export NV_ACC_DEVICE_TYPE=NVIDIA
export NV_ACC_DEVICE_NUM=0
export ACC_DEVICE_NUM=0

############################################################
# 6. CPU / 线程控制（共享机器必备）
############################################################
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

echo "[CPU PIN] CPUSET=${CPUSET}"
echo "[CPU PIN] OMP_NUM_THREADS=${OMP_NUM_THREADS}"

############################################################
# 7. SANITY CHECK（保持原样）
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
# 8. Run
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxf${MAXFUN}_start${START}_str${STRIDE}_recon${RECONOPT}_proxMode${PROXMODE}_${TS}_RH.log"

echo "================ RUN ================="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PROX_CUDA_VISIBLE_DEVICES=${PROX_CUDA_VISIBLE_DEVICES}"
echo "CPUSET=${CPUSET}"
echo "SCRIPT=${PYTHON_SCRIPT}"
echo "LOG_FILE=${LOG_FILE}"
echo "======================================"

# 🔒 用 taskset 包住整个 python（main + prox 都继承）
taskset -c ${CPUSET} \
python -u "${PYTHON_SCRIPT}" \
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
  --prox_mode "${PROXMODE}" \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee "${LOG_FILE}"
