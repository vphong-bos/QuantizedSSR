#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
PY_DEPS_DIR="${REPO_ROOT}/python_env_ssr_pkgs"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"
DATA_DIR="${SSR_DIR}/data"
DATASET_DST="${DATA_DIR}/dataset"

mkdir -p "${PY_DEPS_DIR}"

export PY_DEPS_DIR
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PY_DEPS_DIR}:${REPO_ROOT}:${SSR_DIR}"

echo "Installing PyTorch 2.1 CPU stack..."
python -m pip install -q --target "${PY_DEPS_DIR}" --upgrade --ignore-installed \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.1.0" \
  "torchvision==0.16.0" \
  "torchaudio==2.1.0"

echo "Installing mmcv-full for torch2.1 CPU..."
python -m pip install -q --target "${PY_DEPS_DIR}" --upgrade --ignore-installed \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html \
  "mmcv-full==1.7.2"

echo "Installing mmdet 2.26.0..."
python -m pip install -q --target "${PY_DEPS_DIR}" --upgrade --ignore-installed \
  "mmdet==2.26.0"

echo "Installing the rest..."
python -m pip install -q --target "${PY_DEPS_DIR}" --upgrade --ignore-installed \
  "ptflops" \
  "pydeps" \
  "tach" \
  "colorama" \
  "pycocotools" \
  "pytest-html" \
  "openpyxl" \
  "rich>=14.2.0" \
  "pyquaternion" \
  "flowlib" \
  "torch-fidelity==0.3.0" \
  "torcheval==0.0.7" \
  "torchmetrics==1.3.1" \
  "torchsummary==1.5.1" \
  "accelerate<0.32" \
  "transformers<4.41" \
  "scikit-image==0.21.0" \
  "scikit-learn==1.3.2"

echo "Installing nuscenes-devkit without resolver backtracking..."
python -m pip install -q --target "${PY_DEPS_DIR}" --upgrade --ignore-installed --no-deps \
  "nuscenes-devkit==1.2.0"

echo "Verifying imports..."
python - <<'PY'
import sys, os
sys.path.insert(0, os.environ["PY_DEPS_DIR"])
import torch, torchvision, torchaudio
import mmcv, mmdet

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmcv file:", mmcv.__file__)
PY

echo "Reconstructing data.zip..."
cd "${SSR_DIR}"
rm -f data.zip
cat data.zip.part-a* > data.zip

echo "Unzipping data.zip..."
unzip -o data.zip

echo "Preparing dataset..."
mkdir -p "${DATA_DIR}"

if [[ -d "${DATASET_SRC}" ]]; then
  rm -rf "${DATASET_DST}"
  mkdir -p "${DATASET_DST}"
  cp -r "${DATASET_SRC}/." "${DATASET_DST}/"
  echo "Dataset copied to ${DATASET_DST}"
else
  echo "ERROR: Dataset source not found: ${DATASET_SRC}"
  exit 1
fi

echo "Done."
echo "Run later with:"
echo "export PY_DEPS_DIR=${PY_DEPS_DIR}"
echo "export PYTHONNOUSERSITE=1"
echo "export PYTHONPATH=${PY_DEPS_DIR}:${REPO_ROOT}:${SSR_DIR}"
echo "cd ${SSR_DIR}"
echo "python run.py <args>"