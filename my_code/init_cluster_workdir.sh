#!/usr/bin/env bash
set -euo pipefail

CUR_DIR="$PWD"
if [[ "$(basename "$CUR_DIR")" == "my_code" ]]; then
  TARGET_DIR="$(cd "$CUR_DIR/.." && pwd)"
else
  TARGET_DIR="$CUR_DIR"
fi

PACKAGE_ROOT="/home/bozhen2/Project/fix_versions/new_v2"
REPO_ROOT="/home/bozhen2/Project/fix_versions"
TEMPLATE_DIR="${PACKAGE_ROOT}/my_code"

echo "Initializing CLUSTER workdir at:"
echo "  ${TARGET_DIR}"
echo ""

mkdir -p "${TARGET_DIR}/logs/condor"
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

copy_if_missing "${TEMPLATE_DIR}/main.py"              "${TARGET_DIR}/my_code/main.py"
copy_if_missing "${TEMPLATE_DIR}/config.json"          "${TARGET_DIR}/my_code/config.json"
copy_if_missing "${TEMPLATE_DIR}/cluster_config.json"  "${TARGET_DIR}/my_code/cluster_config.json"
copy_if_missing "${TEMPLATE_DIR}/cluster_run.sh"       "${TARGET_DIR}/my_code/cluster_run.sh"
copy_if_missing "${TEMPLATE_DIR}/cluster.sub"          "${TARGET_DIR}/my_code/cluster.sub"

chmod +x "${TARGET_DIR}/my_code/cluster_run.sh"

TARGET_DIR_ENV="${TARGET_DIR}" PACKAGE_ROOT_ENV="${PACKAGE_ROOT}" REPO_ROOT_ENV="${REPO_ROOT}" python3 - <<'PY'
from pathlib import Path
import json
import os

target_dir = Path(os.environ["TARGET_DIR_ENV"]).resolve()
package_root = Path(os.environ["PACKAGE_ROOT_ENV"]).resolve()
repo_root = Path(os.environ["REPO_ROOT_ENV"]).resolve()

sub_path = target_dir / "my_code" / "cluster.sub"
cfg_path = target_dir / "my_code" / "cluster_config.json"
run_path = target_dir / "my_code" / "cluster_run.sh"

# patch cluster.sub
text = sub_path.read_text()
lines = text.splitlines()

for i, line in enumerate(lines):
    s = line.strip()
    if s.startswith("initialdir"):
        lines[i] = f"initialdir      = {target_dir}"
    elif s.startswith("executable"):
        lines[i] = f"executable      = {target_dir}/my_code/cluster_run.sh"
    elif s.startswith("environment"):
        lines[i] = (
            'environment     = '
            f'"CUDA_DEVICE_ORDER=PCI_BUS_ID;'
            f'NEW_V2_ROOT={package_root};'
            f'REPO_ROOT={repo_root};'
            f'CONFIG_PATH={target_dir}/my_code/cluster_config.json;'
            f'PYTHON_MOD=new_v2.my_code.main;'
            f'WORKDIR={target_dir}"'
        )

sub_path.write_text("\n".join(lines) + "\n")
print("Patched cluster.sub")

# patch cluster_config.json -> worker_script points to package file
data = json.loads(cfg_path.read_text())

def patch_worker_script(obj):
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "worker_script":
                obj[k] = str(package_root / "my_code" / "run_prox_worker.py")
                n += 1
            else:
                n += patch_worker_script(v)
    elif isinstance(obj, list):
        for item in obj:
            n += patch_worker_script(item)
    return n

n = patch_worker_script(data)
cfg_path.write_text(json.dumps(data, indent=2) + "\n")
print(f"Patched cluster_config.json worker_script occurrences: {n}")

# patch cluster_run.sh with absolute paths
run_text = run_path.read_text()
run_text = run_text.replace(
    'CONFIG_PATH="${CONFIG_PATH:-$NEW_V2_ROOT/my_code/cluster_config.json}"',
    f'CONFIG_PATH="${{CONFIG_PATH:-{target_dir}/my_code/cluster_config.json}}"'
)
run_text = run_text.replace(
    'WORKDIR="${WORKDIR:-$PWD}"',
    f'WORKDIR="${{WORKDIR:-{target_dir}}}"'
)
run_text = run_text.replace(
    'LOG_ROOT="${LOG_ROOT:-$WORKDIR/logs}"',
    f'LOG_ROOT="${{LOG_ROOT:-{target_dir}/logs}}"'
)
run_path.write_text(run_text)
print("Patched cluster_run.sh")
PY

echo ""
echo "Done."
echo ""
echo "Run with:"
echo "  cd ${TARGET_DIR}/my_code && condor_submit cluster.sub"
