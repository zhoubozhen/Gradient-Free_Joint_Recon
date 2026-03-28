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

echo "Initializing MPI LOCAL workdir at:"
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

# mpi local 只保留这三个
copy_if_missing "${TEMPLATE_DIR}/main.py"           "${TARGET_DIR}/my_code/main.py"
copy_if_missing "${TEMPLATE_DIR}/mpi_config.json"   "${TARGET_DIR}/my_code/mpi_config.json"
copy_if_missing "${TEMPLATE_DIR}/run_real_mpi.sh"   "${TARGET_DIR}/my_code/run_real_mpi.sh"

chmod +x "${TARGET_DIR}/my_code/run_real_mpi.sh"

TARGET_DIR_ENV="${TARGET_DIR}" PACKAGE_ROOT_ENV="${PACKAGE_ROOT}" python3 - <<'PY'
from pathlib import Path
import json
import os

target_dir = Path(os.environ["TARGET_DIR_ENV"]).resolve()
package_root = Path(os.environ["PACKAGE_ROOT_ENV"]).resolve()

my_code = target_dir / "my_code"
main_py = my_code / "main.py"
mpi_config_json = my_code / "mpi_config.json"
run_real_mpi_sh = my_code / "run_real_mpi.sh"

def patch_json_config(path: Path):
    if not path.exists():
        return
    data = json.loads(path.read_text())

    def walk(obj):
        if isinstance(obj, dict):
            for _, v in obj.items():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Patched {path.name}")

patch_json_config(mpi_config_json)

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

if run_real_mpi_sh.exists():
    text = run_real_mpi_sh.read_text()

    text = text.replace('CONFIG_PATH="${CONFIG_PATH:-my_code/config.json}"',
                        'CONFIG_PATH="${CONFIG_PATH:-my_code/mpi_config.json}"')

    text = text.replace('CONFIG_PATH="${CONFIG_PATH:-config.json}"',
                        'CONFIG_PATH="${CONFIG_PATH:-mpi_config.json}"')

    run_real_mpi_sh.write_text(text)
    print("Patched run_real_mpi.sh (best effort)")
PY

echo ""
echo "Done."
echo ""
echo "Run MPI locally from:"
echo "  cd ${TARGET_DIR}/my_code"
echo "  ./run_real_mpi.sh"