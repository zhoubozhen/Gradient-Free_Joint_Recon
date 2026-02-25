#!/usr/bin/env bash
set -eo pipefail
set -x

############################################################
# 0. Condor execute node PATH 修复（只补，不覆盖）
############################################################
if ! command -v dirname >/dev/null 2>&1; then
  export PATH="/usr/bin:/bin:/usr/local/bin:${PATH:-}"
fi

############################################################
# 1. 参数
############################################################
ENV_NAME=devito
GPU_ID=0

LIP=5
ITER=20
REG=0.0001
MAXFUN=60
RECONOPT=2
STRIDE=1
START=0.98

############################################################
# 2. 路径
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="/home/bozhen2/transcranial_pact_devito_model/V2_tesla"
PYTHON_SCRIPT="${V2_ROOT}/my_code/gfjr_stable.py"

############################################################
# 3. 初始化 module（Condor + Lmod 终极安全写法）
############################################################
set +u

# ★★ 关键修复：防止 Lmod 在 test -z 时被展开炸掉 ★★
export MODULEPATH=""

source /etc/profile.d/modules.sh

set -u

module purge
module use /etc/modulefiles/software
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

############################################################
# 4. Conda
############################################################
source /home/bozhen2/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

############################################################
# 5. GPU / Compiler
############################################################
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export CC=nvc
export CXX=nvc++

############################################################
# 6. Python path
############################################################
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH:-}"

############################################################
# 7. SANITY CHECK（必须全部通过）
############################################################
echo "================ SANITY CHECK ================"
echo "HOST        : $(hostname)"
echo "PATH        : $PATH"

which dirname
which readlink
which python
which nvc++

nvc++ --version

python - <<'PY'
import sys, devito
print("Python =", sys.executable)
print("Devito =", devito.__version__)
PY

echo "============================================="

############################################################
# 8. 日志
############################################################
LOG_DIR="${V2_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/L${LIP}_iter${ITER}_maxf${MAXFUN}_start${START}_str${STRIDE}_recon${RECONOPT}_${TS}_CL.log"

############################################################
# 9. RUN
############################################################
python -u "${PYTHON_SCRIPT}" \
  -i 3 \
  --skullp0 0 \
  --pressure nhp_3_nsp \
  -n 5 \
  -r ${REG} \
  -l ${LIP} \
  --recon_opt ${RECONOPT} \
  --iter ${ITER} \
  --maxfun ${MAXFUN} \
  --stride ${STRIDE} \
  --start ${START} \
  2>&1 | stdbuf -oL -eL \
  grep -v "The -gpu=pinned option is deprecated" \
  | tee "${LOG_FILE}"
