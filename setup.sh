#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "Installing Python requirements on Kaggle..."

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

PIP_USER_FLAG="--user"

python -m pip install ${PIP_USER_FLAG} -q --upgrade pip setuptools wheel packaging

# Remove conflicting OpenMMLab bits if present
python -m pip uninstall -y mmcv mmcv-full mmcv-lite mmengine mmdet mim openmim || true

# Keep Kaggle's existing torch. Do NOT downgrade torch on this runtime.
python - <<'PY'
import sys
try:
    import torch
    print("Using torch:", torch.__version__)
except Exception:
    print("Torch is not installed in this runtime.")
    sys.exit(1)
PY

# Install non-OpenMMLab deps first
REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmengine|mmcv|mmcv-lite|mmcv-full|mmdet|openmim|mim)([<>=!~].*)?$|^[[:space:]]*--find-links ' "${REQ_FILE}" > "${REST_REQ}"
python -m pip install ${PIP_USER_FLAG} -q --prefer-binary -r "${REST_REQ}"
rm -f "${REST_REQ}"

# Install modern OpenMMLab stack without openmim
python -m pip install ${PIP_USER_FLAG} -q --prefer-binary mmengine
python -m pip install ${PIP_USER_FLAG} -q --prefer-binary mmcv-lite
python -m pip install ${PIP_USER_FLAG} -q --prefer-binary mmdet

# Move to working directory (if defined)
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