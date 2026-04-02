#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
PY_DEPS_DIR="${REPO_ROOT}/python_env_ssr_pkgs"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"
DATASET_DST="${SSR_DIR}/data/dataset"

echo "Repo root: ${REPO_ROOT}"
echo "SSR dir: ${SSR_DIR}"
echo "Python deps dir: ${PY_DEPS_DIR}"

cd "${REPO_ROOT}"

mkdir -p "${PY_DEPS_DIR}"

export PY_DEPS_DIR
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PY_DEPS_DIR}:${REPO_ROOT}:${SSR_DIR}"

echo "Installing Python packages..."
python -m pip install -q --target "${PY_DEPS_DIR}" --upgrade --ignore-installed \
  "numpy==2.1.3" \
  "opencv-python>=4.10,<5" \
  "requests>=2.32,<3" \
  "tqdm>=4.67,<5" \
  "filelock>=3.15" \
  "mmengine==0.10.7" \
  "mmcv-lite==2.1.0" \
  "mmdet==3.3.0" \
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
  "nuscenes-devkit==1.2.0" \
  "torch-fidelity==0.3.0" \
  "torcheval==0.0.7" \
  "torchmetrics==1.3.1" \
  "torchsummary==1.5.1" \
  "accelerate<0.32" \
  "transformers<4.41" \
  "scikit-image==0.21.0" \
  "scikit-learn==1.3.2"

echo "Verifying imports..."
python - <<'PY'
import numpy, cv2, mmengine, mmcv, mmdet
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmcv file:", mmcv.__file__)
PY

echo "Preparing dataset..."
mkdir -p "${SSR_DIR}/data"

if [[ -d "${DATASET_SRC}" ]]; then
  echo "Copying dataset..."
  rm -rf "${DATASET_DST}"
  mkdir -p "${DATASET_DST}"
  cp -r "${DATASET_SRC}/." "${DATASET_DST}/"
else
  echo "ERROR: Dataset not found: ${DATASET_SRC}"
  exit 1
fi

echo "Reconstructing data.zip..."
cd "${SSR_DIR}"
rm -f data.zip
cat data.zip.part-a* > data.zip

echo "Unzipping data.zip..."
unzip -o data.zip

echo "Setup complete."
echo ""
echo "To run later:"
echo "----------------------------------------"
echo "export PY_DEPS_DIR=${PY_DEPS_DIR}"
echo "export PYTHONNOUSERSITE=1"
echo "export PYTHONPATH=${PY_DEPS_DIR}:${REPO_ROOT}:${SSR_DIR}"
echo "cd ${SSR_DIR}"
echo "python run.py <your_args>"
echo "----------------------------------------"