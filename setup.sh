#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "Installing Python requirements..."
python -m pip install -q -r "${REQ_FILE}"

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