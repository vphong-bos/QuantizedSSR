#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "Installing Python requirements..."

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

python -m pip install -q --upgrade pip setuptools wheel packaging

# Split OpenMMLab packages from the rest.
REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmengine|mmcv|mmdet)([<>=!~].*)?$' "${REQ_FILE}" > "${REST_REQ}"

# Install everything else first.
python -m pip install -q --prefer-binary -r "${REST_REQ}"

# Install modern OpenMMLab stack without openmim.
python -m pip install -q --prefer-binary mmengine
python -m pip install -q --prefer-binary mmcv
python -m pip install -q --prefer-binary mmdet

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