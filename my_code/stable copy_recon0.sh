#!/usr/bin/env bash
set -euo pipefail

############################################################
# 0. 基本参数
############################################################
ENV_NAME="devito"
GPU_ID="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 1. CUDA module（真实路径已确认）
############################################################
module purge
module load cuda-toolkit/12.2

export CUDA_HOME="/software/cuda-12.2"
export CUDA_PATH="${CUDA_HOME}"           # for some tools
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

############################################################
# 2. NVHPC（只用 compiler；注意：不要让它的 nvcc 抢到 PATH 最前）
############################################################
export NVHPC_ROOT="$HOME/nvhpc"
export NVHPC_VERSION="25.9"

# 先加 NVHPC compiler（会把它自己的 nvcc 带进来）
export PATH="${NVHPC_ROOT}/Linux_x86_64/${NVHPC_VERSION}/compilers/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_ROOT}/Linux_x86_64/${NVHPC_VERSION}/compilers/lib:${LD_LIBRARY_PATH}"

# 再把系统 CUDA/bin 重新放回 PATH 最前，确保 which nvcc = /software/cuda-12.2/bin/nvcc
export PATH="${CUDA_HOME}/bin:${PATH}"

# 让 nvc++ 也认系统 CUDA（避免它回去找自带 CUDA）
export NVCOMPILER_CUDA_HOME="${CUDA_HOME}"
export NVHPC_CUDA_HOME="${CUDA_HOME}"

export CC=nvc
export CXX=nvc++

############################################################
# 3. Conda
############################################################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

############################################################
# 4. Devito / OpenACC（Ada = sm_89）
############################################################
export DEVITO_LANGUAGE=openacc
export DEVITO_PLATFORM=nvidiaX
export DEVITO_COMPILER=nvc

export NVCOMPILER_ACC_GPU=cc89
export DEVITO_ARCH_FLAGS="-gpu=cc89,mem:separate:pinnedalloc"

############################################################
# 5. ★ CuPy / NVRTC 必杀：强制 CUDA 根目录 + include ★
############################################################
# CuPy 用这个确定 CUDA_ROOT（比猜 nvcc 位置更可靠）
export CUPY_CUDA_PATH="${CUDA_HOME}"

# NVRTC 强制 include（防止 cuda_fp16.h 找不到）
export CUPY_NVRTC_OPTIONS="-I${CUDA_HOME}/include"

# 兜底
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

############################################################
# 6. PYTHONPATH
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"

############################################################
# 7. GPU 绑定
############################################################
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

############################################################
# 8. SANITY CHECK（关键：这里必须看到 nvcc 在 /software/cuda-12.2/bin）
############################################################
echo "================ SANITY CHECK ================"
echo "[CUDA_HOME] ${CUDA_HOME}"
echo "[CUPY_CUDA_PATH] ${CUPY_CUDA_PATH}"
echo "[CUPY_NVRTC_OPTIONS] ${CUPY_NVRTC_OPTIONS}"

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
# 9. 清 cache（必须）
############################################################
rm -rf ~/.cache/devito
rm -rf ~/.cupy/kernel_cache
rm -rf /tmp/devito-*
rm -rf /tmp/cupy-*

############################################################
# 10. Run
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/gfjr_gpu_${TS}.log"

echo "================ RUN (GPU, NO MPI) ================"
echo "GPU=${CUDA_VISIBLE_DEVICES}"
echo "SCRIPT=${PYTHON_SCRIPT}"
echo "LOG=${LOG_FILE}"
echo "==================================================="
export NVCOMPILER_FLAGS="-gpu=mem:separate:pinnedalloc"

python -u "${PYTHON_SCRIPT}" \
  -i 3 \
  --skullp0 0 \
  --pressure nhp_3_nsp \
  -n 5 \
  -r 0.0001 \
  -l 5 \
  --recon_opt 0 \
  --iter 1 \
  2>&1 | tee "${LOG_FILE}"
