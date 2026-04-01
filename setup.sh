#!/usr/bin/env bash
set -euo pipefail

# This script expects the shell to already have:
#   source env_set.sh ssr
# so that PYTHON_ENV_DIR / WORKING_DIR / TT_METAL_HOME are set.

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  echo "ERROR: TT_METAL_HOME is not set."
  echo "Run: source env_set.sh ssr"
  exit 1
fi

if [[ -z "${PYTHON_ENV_DIR:-}" ]]; then
  echo "ERROR: PYTHON_ENV_DIR is not set."
  echo "Run: source env_set.sh ssr"
  exit 1
fi

if [[ ! -f "${PYTHON_ENV_DIR}/bin/activate" ]]; then
  echo "ERROR: Python environment not found at: ${PYTHON_ENV_DIR}"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "Activating Python environment: ${PYTHON_ENV_DIR}"
# shellcheck disable=SC1090
source "${PYTHON_ENV_DIR}/bin/activate"

echo "Python executable: $(which python)"
python -c "import sys; print('Python version:', sys.version)"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

echo "Upgrading pip tooling..."
python -m pip install -q --upgrade pip setuptools wheel

echo "Removing conflicting OpenMMLab packages if present..."
python -m pip uninstall -y mmcv mmcv-full mmcv-lite mmengine mmdet mim openmim || true

echo "Checking torch in this environment..."
python - <<'PY'
import sys
try:
    import torch
    print("Torch found:", torch.__version__)
except Exception as e:
    print("Torch is not installed or not importable:", e)
    sys.exit(1)
PY

# Install non-OpenMMLab requirements first
REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmengine|mmcv|mmcv-lite|mmcv-full|mmdet|openmim|mim)([<>=!~].*)?$|^[[:space:]]*--find-links ' "${REQ_FILE}" > "${REST_REQ}"

echo "Installing non-OpenMMLab requirements..."
python -m pip install -q --prefer-binary -r "${REST_REQ}"
rm -f "${REST_REQ}"

echo "Installing OpenMMLab packages..."
python -m pip install -q --prefer-binary mmengine
python -m pip install -q --prefer-binary mmcv-lite
python -m pip install -q --prefer-binary mmdet

echo "Verifying core imports..."
python - <<'PY'
import torch
import mmengine
import mmcv
import mmdet
print("torch:", torch.__version__)
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
PY

# Move to working directory if provided
if [[ -n "${WORKING_DIR:-}" ]]; then
  echo "Changing to WORKING_DIR: $WORKING_DIR"
  cd "$WORKING_DIR"
fi

# Paths
SSR_DIR="${SCRIPT_DIR}/ssr"
REFERENCE_DIR="${SSR_DIR}/reference"
DATA_DIR="${SCRIPT_DIR}/data"
KAGGLE_DATASET_DIR="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"

echo "Preparing data directory..."
mkdir -p "$DATA_DIR"

echo "Reconstructing data.zip from parts..."
cat "${SSR_DIR}"/data.zip.part-a* > "${SSR_DIR}/data.zip"

echo "Unzipping data.zip to data directory..."
unzip -q "${SSR_DIR}/data.zip" -d "$DATA_DIR"

mkdir -p "${DATA_DIR}/dataset"

if [[ -d "$KAGGLE_DATASET_DIR" ]]; then
  echo "Kaggle dataset found. Copying to ${DATA_DIR}/dataset ..."
  cp -r "${KAGGLE_DATASET_DIR}/." "${DATA_DIR}/dataset/"
  echo "Dataset copied successfully."
else
  echo "WARNING: Kaggle dataset not found at ${KAGGLE_DATASET_DIR}"
fi

mkdir -p "$REFERENCE_DIR"
ln -sfn "${DATA_DIR}/dataset" "${REFERENCE_DIR}/dataset"
echo "Symlink created: ${REFERENCE_DIR}/dataset -> ${DATA_DIR}/dataset"

echo "Setup complete."