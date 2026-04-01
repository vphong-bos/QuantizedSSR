#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "Installing Python requirements..."

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

python -m pip install -q --upgrade pip setuptools wheel

# Strongly recommend Python 3.10 for this legacy OpenMMLab stack
PY_VER="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$PY_VER" != "3.10" ]]; then
  echo "ERROR: This environment uses Python ${PY_VER}."
  echo "For mmdet==2.26.0 + mmcv-full==1.7.2, use Python 3.10."
  exit 1
fi

# Install a legacy-compatible PyTorch first.
# CPU build:
python -m pip install -q --prefer-binary \
  torch==1.13.1 torchvision==0.14.1 \
  --index-url https://download.pytorch.org/whl/cpu

# Install matching OpenMMLab legacy stack.
# Do NOT install mmcv and mmcv-full together.
python -m pip install -q --prefer-binary \
  mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html

python -m pip install -q --prefer-binary \
  mmdet==2.26.0

# Install the rest, excluding conflicting OpenMMLab lines
REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmcv-full|mmcv|mmengine|mmdet)==|^[[:space:]]*--find-links ' "${REQ_FILE}" > "${REST_REQ}"

python -m pip install -q --prefer-binary -r "${REST_REQ}"
rm -f "${REST_REQ}"

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
  echo "Please follow the instructions here to download and prepare the dataset:"
  echo "https://github.com/bos-semi/tt-metal/blob/develop/models/bos_model/ssr/README.md"
fi

mkdir -p "$REFERENCE_DIR"
ln -sfn "${DATA_DIR}/dataset" "${REFERENCE_DIR}/dataset"
echo "Symlink created: ${REFERENCE_DIR}/dataset -> ${DATA_DIR}/dataset"

echo "Setup complete."