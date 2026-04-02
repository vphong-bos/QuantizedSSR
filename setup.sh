#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"

echo "Preparing SSR directory..."
mkdir -p "${SSR_DIR}"

echo "Reconstructing data.zip..."
cd "${SSR_DIR}"
rm -f data.zip
cat data.zip.part-a* > data.zip

echo "Unzipping data.zip..."
unzip -o data.zip

echo "Preparing dataset directory..."
DATA_DIR="${SSR_DIR}/data"
DATASET_DST="${DATA_DIR}/dataset"

mkdir -p "${DATA_DIR}"

if [[ -d "${DATASET_SRC}" ]]; then
  echo "Copying dataset..."
  rm -rf "${DATASET_DST}"
  mkdir -p "${DATASET_DST}"
  cp -r "${DATASET_SRC}/." "${DATASET_DST}/"
  echo "Dataset copied to ${DATASET_DST}"
else
  echo "ERROR: Dataset source not found: ${DATASET_SRC}"
  exit 1
fi

echo "Done."