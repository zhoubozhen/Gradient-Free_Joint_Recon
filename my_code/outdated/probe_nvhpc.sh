#!/usr/bin/env bash
set -o pipefail
set -x

echo "HOST=$(hostname)"
echo "SHELL=$SHELL"
echo "PATH(before)=$PATH"

# ---- module init (CRITICAL) ----
set +u
source /etc/profile.d/modules.sh
set -u
# --------------------------------

module purge
module load nvidia-hpc-sdk-multi/25.1-rh8
module load cuda-toolkit/12.2

echo "PATH(after)=$PATH"

type module
which nvc || true
which nvc++ || true
nvc++ --version || true

python - <<'PY'
import os
print("CC =", os.environ.get("CC"))
print("CXX =", os.environ.get("CXX"))
print("PATH =", os.environ.get("PATH"))
PY
