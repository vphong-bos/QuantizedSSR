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

# Install all Python packages into an isolated local dir instead of touching Kaggle system packages.
export PY_DEPS_DIR="${TT_METAL_HOME}/python_env_ssr_pkgs"
mkdir -p "${PY_DEPS_DIR}"

export PYTHONPATH="${PY_DEPS_DIR}:${TT_METAL_HOME}:${BOS_METAL_HOME}:${WORKING_DIR}:SSR:${PYTHONPATH:-}"

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
print("PYTHONPATH:", sys.path[:5])
PY

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

TARGET_FLAG=(--target "${PY_DEPS_DIR}" --upgrade --ignore-installed)

echo "Installing isolated core stack into ${PY_DEPS_DIR} ..."
python -m pip install -q "${TARGET_FLAG[@]}" \
  "numpy==2.1.3" \
  "requests>=2.32,<3" \
  "tqdm>=4.67,<5" \
  "filelock>=3.15" \
  "opencv-python>=4.10"

echo "Checking torch from base runtime..."
python - <<'PY'
import sys
try:
    import torch
    print("Torch found:", torch.__version__)
except Exception as e:
    print("Torch is not installed or not importable:", e)
    sys.exit(1)
PY

REST_REQ="$(mktemp)"
grep -vE '^[[:space:]]*(mmengine|mmcv|mmcv-lite|mmcv-full|mmdet|openmim|mim)([<>=!~].*)?$|^[[:space:]]*--find-links ' "${REQ_FILE}" > "${REST_REQ}"

echo "Installing non-OpenMMLab requirements into isolated dir..."
python -m pip install -q "${TARGET_FLAG[@]}" -r "${REST_REQ}"
rm -f "${REST_REQ}"

echo "Installing OpenMMLab packages into isolated dir..."
python -m pip install -q "${TARGET_FLAG[@]}" \
  "mmengine>=0.10.7,<1.0.0" \
  "mmcv-lite>=2.1.0,<2.3.0" \
  "mmdet>=3.3.0,<3.4.0"

echo "Verifying imports from isolated dir..."
python - <<'PY'
import sys
print("site path head:", sys.path[:8])

import numpy
import cv2
import torch
import mmengine
import mmcv
import mmdet

print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("torch:", torch.__version__)
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
echo "Isolated packages directory: ${PY_DEPS_DIR}"