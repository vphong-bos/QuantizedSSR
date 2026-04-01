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
export PY_DEPS_DIR="${TT_METAL_HOME}/python_env_ssr_pkgs"

mkdir -p "${PY_DEPS_DIR}"

# 👇 CRITICAL: force Python to use our packages
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PY_DEPS_DIR}:${TT_METAL_HOME}:${BOS_METAL_HOME}:${WORKING_DIR}:SSR"

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
print("sys.path head:", sys.path[:8])
PY

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_PROGRESS_BAR=off
export PIP_PREFER_BINARY=1

TARGET_FLAG=(--target "${PY_DEPS_DIR}" --upgrade --ignore-installed)

echo "Installing core dependencies..."
python -m pip install -q "${TARGET_FLAG[@]}" \
  "numpy==2.1.3" \
  "requests>=2.32,<3" \
  "tqdm>=4.67,<5" \
  "filelock>=3.15" \
  "opencv-python>=4.10,<5"

echo "Checking torch..."
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

echo "Verifying..."
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

echo "Setup complete."