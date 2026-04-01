#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "Installing Python requirements..."

# Prefer wheels and reuse cache
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

python -m pip install -q --upgrade pip setuptools wheel

# mmcv-full already covers mmcv use cases here, so skip plain mmcv to avoid duplicate work/conflicts.
TMP_REQ="$(mktemp)"
awk '
  BEGIN { skip_mmcv=0 }
  /^[[:space:]]*mmcv-full==/ { skip_mmcv=1; print; next }
  /^[[:space:]]*mmcv==/ {
    if (skip_mmcv == 1) next
  }
  { print }
' "${REQ_FILE}" > "${TMP_REQ}"

# Install OpenMMLab pieces first, forcing prebuilt wheels from the provided index.
python -m pip install -q --prefer-binary \
  --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html \
  mmcv-full==1.7.2 \
  mmengine==0.10.7 \
  mmdet==2.26.0

# Install the rest from requirements, excluding lines already handled above.
REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmcv-full|mmcv|mmengine|mmdet)==|^[[:space:]]*--find-links ' "${TMP_REQ}" > "${REST_REQ}"

python -m pip install -q --prefer-binary -r "${REST_REQ}"

rm -f "${TMP_REQ}" "${REST_REQ}"

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

# Ensure dataset folder exists inside data
mkdir -p "${DATA_DIR}/dataset"

# Try to copy dataset from Kaggle
if [[ -d "$KAGGLE_DATASET_DIR" ]]; then
  echo "Kaggle dataset found. Copying to ${DATA_DIR}/dataset ..."
  cp -r "${KAGGLE_DATASET_DIR}/." "${DATA_DIR}/dataset/"
  echo "Dataset copied successfully."
else
  echo "WARNING: Kaggle dataset not found at ${KAGGLE_DATASET_DIR}"
  echo "Please follow the instructions here to download and prepare the dataset:"
  echo "https://github.com/bos-semi/tt-metal/blob/develop/models/bos_model/ssr/README.md"
fi

# Create symlink so ssr/reference can use the dataset too
mkdir -p "$REFERENCE_DIR"
ln -sfn "${DATA_DIR}/dataset" "${REFERENCE_DIR}/dataset"
echo "Symlink created: ${REFERENCE_DIR}/dataset -> ${DATA_DIR}/dataset"

echo "Setup complete."