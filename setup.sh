#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
PY_DEPS_DIR="${SCRIPT_DIR}/python_env_ssr_pkgs"

cd "${REPO_ROOT}"

echo "Setting Python environment..."

mkdir -p "${PY_DEPS_DIR}"

export PY_DEPS_DIR
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PY_DEPS_DIR}:${REPO_ROOT}:${REPO_ROOT}/ssr:${PYTHONPATH:-}"

echo "Python executable: $(which python)"
python - <<'PY'
import sys, os
print("Python version:", sys.version)
print("PY_DEPS_DIR:", os.environ.get("PY_DEPS_DIR"))
print("sys.path head:", sys.path[:6])
PY

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

TARGET_FLAG=(--target "${PY_DEPS_DIR}" --upgrade --ignore-installed)

echo "Installing core deps into ${PY_DEPS_DIR} ..."
python -m pip install -q "${TARGET_FLAG[@]}" \
  "numpy==2.1.3" \
  "requests>=2.32,<3" \
  "tqdm>=4.67,<5" \
  "filelock>=3.15" \
  "opencv-python>=4.10,<5"

echo "Checking torch from base runtime..."
python - <<'PY'
import torch
print("Torch:", torch.__version__)
PY

echo "Installing project requirements..."
python -m pip install -q "${TARGET_FLAG[@]}" -r "${REQ_FILE}"

echo "Installing OpenMMLab..."
python -m pip install -q "${TARGET_FLAG[@]}" \
  "mmengine==0.10.7" \
  "mmcv-lite==2.1.0" \
  "mmdet==3.3.0"

echo "Verifying imports..."
python - <<'PY'
import sys
import numpy, cv2, torch, mmengine, mmcv, mmdet
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("torch:", torch.__version__)
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmcv path:", mmcv.__file__)
PY

echo "Writing run helper..."
cat > "${SCRIPT_DIR}/run_ssr.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PY_DEPS_DIR="${PY_DEPS_DIR}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PY_DEPS_DIR}:${REPO_ROOT}:${REPO_ROOT}/ssr"
cd "${REPO_ROOT}"
python ssr/run.py "\$@"
EOF
chmod +x "${SCRIPT_DIR}/run_ssr.sh"

echo "Setup complete."
echo "Run with: ${SCRIPT_DIR}/run_ssr.sh --help"