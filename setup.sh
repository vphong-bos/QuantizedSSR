#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

cd "${REPO_ROOT}"

echo "Setting SSR environment variables..."

export ARCH_NAME=blackhole
export TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS="BR,NC,TR,ER"
export TT_METAL_HOME="${REPO_ROOT}"
export WORKING_DIR="${TT_METAL_HOME}/models/bos_model/ssr"
export BOS_METAL_HOME="${TT_METAL_HOME}/tt_metal/third_party/bos-metal"
export PYTHONPATH="${TT_METAL_HOME}:${BOS_METAL_HOME}:${WORKING_DIR}:SSR:${PYTHONPATH:-}"

if [[ "${TT_METAL_ENABLE_DEBUG:-0}" -eq 1 ]]; then
  export TT_METAL_LOGGER_LEVEL="Debug"
  export TT_METAL_LOGGER_TYPES="Op"
  export TT_METAL_DPRINT_CHIPS=0
  export TT_METAL_DPRINT_CORES=0,0
  export TTNN_TILIZE_FORCE_SINGLE_TILE_INTERLEAVED=1
fi

echo "Python executable: $(which python)"
python - <<'PY'
import sys
print("Python version:", sys.version)
PY

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off

echo "Upgrading pip..."
python -m pip install -q --upgrade pip setuptools wheel

echo "Installing requirements..."
python -m pip install -q --no-deps -r "${REQ_FILE}"

# Move to working directory
echo "Changing to WORKING_DIR: ${WORKING_DIR}"
cd "${WORKING_DIR}"

# Paths
SSR_DIR="${SCRIPT_DIR}/ssr"
REFERENCE_DIR="${SSR_DIR}/reference"
DATA_DIR="${SCRIPT_DIR}/data"
KAGGLE_DATASET_DIR="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"

echo "Preparing data directory..."
mkdir -p "${DATA_DIR}"

echo "Reconstructing data.zip from parts..."
cat "${SSR_DIR}"/data.zip.part-a* > "${SSR_DIR}/data.zip"

echo "Unzipping data.zip to data directory..."
unzip -q "${SSR_DIR}/data.zip" -d "${DATA_DIR}"

mkdir -p "${DATA_DIR}/dataset"

if [[ -d "${KAGGLE_DATASET_DIR}" ]]; then
  echo "Kaggle dataset found. Copying to ${DATA_DIR}/dataset ..."
  cp -r "${KAGGLE_DATASET_DIR}/." "${DATA_DIR}/dataset/"
  echo "Dataset copied successfully."
else
  echo "WARNING: Kaggle dataset not found at ${KAGGLE_DATASET_DIR}"
fi

mkdir -p "${REFERENCE_DIR}"
ln -sfn "${DATA_DIR}/dataset" "${REFERENCE_DIR}/dataset"
echo "Symlink created: ${REFERENCE_DIR}/dataset -> ${DATA_DIR}/dataset"

echo "Setup complete."