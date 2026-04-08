#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"
DATA_DIR="${SSR_DIR}/data"
DATASET_DST="${DATA_DIR}/dataset"
PY310_ENV="/kaggle/working/py310_ssr"

echo "Installing Python 3.10..."
apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y python3.10 python3.10-distutils python3.10-venv

echo "Installing pip for Python 3.10..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3.10 /tmp/get-pip.py

echo "Creating virtual environment..."
rm -rf "${PY310_ENV}"
python3.10 -m venv "${PY310_ENV}"
source "${PY310_ENV}/bin/activate"

echo "Installing base packaging tools..."
python -m pip install -U "pip<25" "setuptools<81" wheel

python -m pip install lazy_loader

python -m pip install aimet-torch

python -m pip install onnxruntime-gpu==1.23.0

python -m pip install onnx onnxruntime onnxruntime-extensions

python -m pip install joblib threadpoolctl scipy tqdm cachetools loguru shapely descartes ipython seaborn

echo "Installing legacy-compatible opencv..."
python -m pip install --force-reinstall --no-deps "opencv-python-headless<4.13"

echo "Installing PyTorch 2.1 CPU stack..."
python -m pip install \
  torch==2.1.0 \
  torchvision==0.16.0 \
  torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cpu

echo "Installing MMCV/MMDetection legacy stack..."
python -m pip install \
  mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

python -m pip install mmdet==2.26.0

echo "Installing remaining dependencies..."
python -m pip install \
  ptflops \
  pydeps \
  tach \
  colorama \
  pycocotools \
  pytest-html \
  openpyxl \
  "rich>=14.2.0" \
  pyquaternion \
  flowlib \
  nuscenes-devkit==1.2.0 \
  torch-fidelity==0.3.0 \
  torcheval==0.0.7 \
  torchmetrics==1.3.1 \
  torchsummary==1.5.1 \
  "accelerate<0.32" \
  "transformers<4.41" \
  scikit-image==0.21.0 \
  scikit-learn==1.3.2 \
  --no-deps

echo "Installing extra deps..."
python -m pip install \
  "matplotlib>=3.6" \
  "pillow" \
  "packaging" \
  "pyyaml" \
  --no-deps

echo "Force reinstalling NumPy (final step)..."

# uninstall ANY existing numpy (ignore errors)
python -m pip uninstall -y numpy || true

# install exact version without dependency interference
python -m pip install --no-cache-dir --no-deps numpy==1.26.4

echo "Verifying NumPy..."
python - <<'PY'
import numpy, sys
print("python exe:", sys.executable)
print("numpy version:", numpy.__version__)
print("numpy path:", numpy.__file__)
assert numpy.__version__ == "1.26.4", f"Expected 1.26.4, got {numpy.__version__}"
PY

echo "Verifying environment..."
python - <<'PY'
import sys
import pkg_resources
import numpy
import cv2
import torch, torchvision, torchaudio
import mmcv, mmdet

print("python:", sys.version)
print("setuptools/pkg_resources ok")
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmcv path:", mmcv.__file__)
PY

echo "Done."