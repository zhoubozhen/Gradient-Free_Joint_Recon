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

TEMPLATE_DIR="${PACKAGE_ROOT}/my_code"

echo "Initializing LOCAL workdir at:"
echo "  ${TARGET_DIR}"
echo ""
echo "PACKAGE_ROOT=${PACKAGE_ROOT}"
echo ""

mkdir -p "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/logs"
mkdir -p "${TARGET_DIR}/logs/condor"
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

copy_if_missing "${TEMPLATE_DIR}/main.py"              "${TARGET_DIR}/my_code/main.py"
copy_if_missing "${TEMPLATE_DIR}/config.json"          "${TARGET_DIR}/my_code/config.json"
copy_if_missing "${TEMPLATE_DIR}/cluster_config.json"  "${TARGET_DIR}/my_code/cluster_config.json"
copy_if_missing "${TEMPLATE_DIR}/cluster_run.sh"       "${TARGET_DIR}/my_code/cluster_run.sh"
copy_if_missing "${TEMPLATE_DIR}/cluster.sub"          "${TARGET_DIR}/my_code/cluster.sub"
copy_if_missing "${TEMPLATE_DIR}/run_prox_worker.py"   "${TARGET_DIR}/my_code/run_prox_worker.py"
copy_if_missing "${TEMPLATE_DIR}/run_local.sh"          "${TARGET_DIR}/my_code/run_local.sh"

chmod +x "${TARGET_DIR}/my_code/cluster_run.sh"
chmod +x "${TARGET_DIR}/my_code/run_local.sh"

TARGET_DIR_ENV="${TARGET_DIR}" PACKAGE_ROOT_ENV="${PACKAGE_ROOT}" python3 - <<'PY'
from pathlib import Path
import json
import os
import re

target_dir = Path(os.environ["TARGET_DIR_ENV"]).resolve()
package_root = Path(os.environ["PACKAGE_ROOT_ENV"]).resolve()

my_code = target_dir / "my_code"
main_py = my_code / "main.py"
config_json = my_code / "config.json"
cluster_config_json = my_code / "cluster_config.json"
cluster_run_sh = my_code / "cluster_run.sh"
cluster_sub = my_code / "cluster.sub"

if cluster_sub.exists():
    text = cluster_sub.read_text()
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
                f'NEW_V2_ROOT={target_dir};'
                f'REPO_ROOT={target_dir};'
                f'CONFIG_PATH={target_dir}/my_code/cluster_config.json;'
                f'PYTHON_MOD=my_code.main;'
                f'WORKDIR={target_dir}"'
            )
    cluster_sub.write_text("\n".join(lines) + "\n")
    print("Patched cluster.sub")

if cluster_run_sh.exists():
    text = cluster_run_sh.read_text()
    text = re.sub(r'NEW_V2_ROOT="\$\{NEW_V2_ROOT:-.*?\}"', f'NEW_V2_ROOT="${{NEW_V2_ROOT:-{target_dir}}}"', text)
    text = re.sub(r'REPO_ROOT="\$\{REPO_ROOT:-.*?\}"',   f'REPO_ROOT="${{REPO_ROOT:-{target_dir}}}"', text)
    text = re.sub(r'PYTHON_MOD="\$\{PYTHON_MOD:-.*?\}"', 'PYTHON_MOD="${PYTHON_MOD:-my_code.main}"', text)
    text = re.sub(r'CONFIG_PATH="\$\{CONFIG_PATH:-.*?\}"', f'CONFIG_PATH="${{CONFIG_PATH:-{target_dir}/my_code/config.json}}"', text)
    text = re.sub(r'WORKDIR="\$\{WORKDIR:-.*?\}"', f'WORKDIR="${{WORKDIR:-{target_dir}}}"', text)
    text = re.sub(r'LOG_ROOT="\$\{LOG_ROOT:-.*?\}"', f'LOG_ROOT="${{LOG_ROOT:-{target_dir}/logs}}"', text)
    cluster_run_sh.write_text(text)
    print("Patched cluster_run.sh")

def patch_json_config(path: Path):
    if not path.exists():
        return
    data = json.loads(path.read_text())

    def walk(obj):
        changed = 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "worker_script":
                    obj[k] = f"{package_root}/my_code/run_prox_worker.py"
                    changed += 1
                else:
                    changed += walk(v)
        elif isinstance(obj, list):
            for item in obj:
                changed += walk(item)
        return changed

    n = walk(data)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Patched {path.name}: worker_script occurrences = {n}")

patch_json_config(config_json)
patch_json_config(cluster_config_json)

if main_py.exists():
    text = main_py.read_text()
    text = text.replace("new_v2.my_code", "my_code")
    package_src = str(package_root / "src")
    text = text.replace(
        '    script_dir = os.path.dirname(os.path.abspath(__file__))\n'
        '    package_root = os.path.dirname(script_dir)\n'
        '    src_dir = os.path.join(package_root, "src")\n'
        '    if src_dir not in sys.path:\n'
        '        sys.path.insert(0, src_dir)\n',
        f'    src_dir = r"{package_src}"\n'
        '    if src_dir not in sys.path:\n'
        '        sys.path.insert(0, src_dir)\n'
    )
    main_py.write_text(text)
    print("Patched main.py (best effort)")
PY

echo ""
echo "Done."
echo ""
echo "Run locally from:"
echo "  cd ${TARGET_DIR}/my_code"
echo "  python3 -u main.py --config config.json"
