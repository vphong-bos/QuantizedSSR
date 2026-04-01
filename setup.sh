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
export PYTHONPATH="${TT_METAL_HOME}:${BOS_METAL_HOME}:${PYTHONPATH:-}:${WORKING_DIR}:SSR"

if [[ "${TT_METAL_ENABLE_DEBUG:-0}" -eq 1 ]]; then
  export TT_METAL_LOGGER_LEVEL="Debug"
  export TT_METAL_LOGGER_TYPES="Op"
  export TT_METAL_DPRINT_CHIPS=0
  export TT_METAL_DPRINT_CORES=0,0
  export TTNN_TILIZE_FORCE_SINGLE_TILE_INTERLEAVED=1
fi

PYTHON_ENV_DIR="${TT_METAL_HOME}/python_env_ssr"
if [[ -f "${PYTHON_ENV_DIR}/bin/activate" ]]; then
  echo "Activating existing Python environment: ${PYTHON_ENV_DIR}"
  # shellcheck disable=SC1090
  source "${PYTHON_ENV_DIR}/bin/activate"
else
  echo "python_env_ssr not found, using current Python environment."
fi

echo "Python executable: $(which python)"
python - <<'PY'
import sys
print("Python version:", sys.version)
PY

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

echo "Upgrading pip tooling..."
python -m pip install -q --upgrade pip setuptools wheel

echo "Removing conflicting OpenMMLab packages if present..."
python -m pip uninstall -y mmcv mmcv-full mmcv-lite mmengine mmdet mim openmim || true

echo "Checking torch in current environment..."
python - <<'PY'
import sys
try:
    import torch
    print("Torch found:", torch.__version__)
except Exception as e:
    print("Torch is not installed or not importable:", e)
    sys.exit(1)
PY

echo "Fixing Kaggle core dependencies..."
python -m pip install -q --upgrade \
  "numpy>=2.0" \
  "requests>=2.32" \
  "tqdm>=4.67" \
  "filelock>=3.15" \
  "opencv-python>=4.13"

REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmengine|mmcv|mmcv-lite|mmcv-full|mmdet|openmim|mim)([<>=!~].*)?$|^[[:space:]]*--find-links ' "${REQ_FILE}" > "${REST_REQ}"

echo "Installing non-OpenMMLab requirements..."
python -m pip install -q --prefer-binary --no-deps -r "${REST_REQ}"
rm -f "${REST_REQ}"

echo "Installing OpenMMLab packages..."
python -m pip install -q --prefer-binary mmengine
python -m pip install -q --prefer-binary mmcv-lite
python -m pip install -q --prefer-binary mmdet

echo "Verifying core imports..."
python - <<'PY'
import numpy
import torch
import cv2
import mmengine
import mmcv
import mmdet

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cv2:", cv2.__version__)
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
PY

echo "Changing to WORKING_DIR: ${WORKING_DIR}"
cd "${WORKING_DIR}"

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