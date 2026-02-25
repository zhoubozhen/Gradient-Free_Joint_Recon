#!/usr/bin/env bash
set -euo pipefail

############################################################
# 0. 基本参数
############################################################
ENV_NAME="devito"
GPU_ID="0"

# --------- 核心实验参数（单一真源）---------
LIP=6
ITER=20
REG=0.0001
MAXFUN=60
RECONOPT=2
STRIDE=$(python -c 'print(4/3)')

# -------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 1. CUDA module
############################################################
module purge
module load cuda-toolkit/12.2

export CUDA_HOME="/software/cuda-12.2"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

############################################################
# 2. NVHPC（compiler only）
############################################################
export NVHPC_ROOT="$HOME/nvhpc"
export NVHPC_VERSION="25.9"

export PATH="${NVHPC_ROOT}/Linux_x86_64/${NVHPC_VERSION}/compilers/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_ROOT}/Linux_x86_64/${NVHPC_VERSION}/compilers/lib:${LD_LIBRARY_PATH}"

# 确保系统 CUDA 的 nvcc 在最前
export PATH="${CUDA_HOME}/bin:${PATH}"

export NVCOMPILER_CUDA_HOME="${CUDA_HOME}"
export NVHPC_CUDA_HOME="${CUDA_HOME}"

export CC=nvc
export CXX=nvc++

############################################################
# 3. ★ 彻底解决 -gpu=pinned 的关键设置 ★
############################################################
export DEVITO_OPENACC_FLAGS="-gpu=cc89,mem:separate:pinnedalloc"
export DEVITO_ARCH_FLAGS="-gpu=cc89,mem:separate:pinnedalloc"
export NVCOMPILER_FLAGS="-gpu=mem:separate:pinnedalloc"

############################################################
# 4. Conda
############################################################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

############################################################
# 5. Devito / OpenACC
############################################################
export DEVITO_LANGUAGE=openacc
export DEVITO_PLATFORM=nvidiaX
export DEVITO_COMPILER=nvc
export NVCOMPILER_ACC_GPU=cc89

############################################################
# 6. CuPy / NVRTC（防 include 坑）
############################################################
export CUPY_CUDA_PATH="${CUDA_HOME}"
export CUPY_NVRTC_OPTIONS="-I${CUDA_HOME}/include"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

############################################################
# 7. PYTHONPATH
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"

############################################################
# 8. GPU 绑定
############################################################
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

############################################################
# 9. SANITY CHECK
############################################################
echo "================ SANITY CHECK ================"
echo "[CUDA_HOME]              ${CUDA_HOME}"
echo "[DEVITO_OPENACC_FLAGS]   ${DEVITO_OPENACC_FLAGS}"
echo "[DEVITO_ARCH_FLAGS]      ${DEVITO_ARCH_FLAGS}"
echo "[NVCOMPILER_FLAGS]       ${NVCOMPILER_FLAGS}"
echo "[PARAM] LIP=${LIP} ITER=${ITER} REG=${REG}"

which nvcc
readlink -f "$(which nvcc)"
nvcc --version | head -n 5

which nvc++
nvc++ --version | head -n 5

which python
python - << 'EOF'
import devito, cupy, sys
print("Devito:", devito.__version__)
print("CuPy:", cupy.__version__)
print("Python:", sys.executable)
EOF

nvidia-smi
echo "=============================================="

############################################################
# 10. 清 cache（必须）
############################################################
rm -rf ~/.cache/devito
rm -rf ~/.cupy/kernel_cache
rm -rf /tmp/devito-*
rm -rf /tmp/cupy-*
TIMESTAMP="$(date '+%Y%m%d_%H%M')"

############################################################
# 11. Run
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_reg${REG}.log"
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxfun${MAXFUN}_recon${RECONOPT}_${TIMESTAMP}_tesla.log"

echo "================ RUN (GPU, NO MPI) ================"
echo "GPU=${CUDA_VISIBLE_DEVICES}"
echo "SCRIPT=${PYTHON_SCRIPT}"
echo "LOG=${LOG_FILE}"
echo "==================================================="

python -u "${PYTHON_SCRIPT}" \
  -i 3 \
  --skullp0 0 \
  --pressure nhp_3_nsp_3ppw \
  -n 5 \
  -r "${REG}" \
  -l "${LIP}" \
  --recon_opt "${RECONOPT}" \
  --iter "${ITER}" \
  --maxfun "${MAXFUN}"\
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee "${LOG_FILE}"
