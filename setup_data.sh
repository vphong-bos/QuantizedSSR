#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"
DATA_DIR="${SSR_DIR}/data"
DATASET_DST="${DATA_DIR}/dataset"
PY39_ENV="/kaggle/working/py39_ssr"

echo "Preparing SSR directory..."
mkdir -p "${SSR_DIR}"

echo "Reconstructing data.zip..."
cd "${SSR_DIR}"
rm -f data.zip
cat data.zip.part-a* > data.zip

echo "Unzipping data.zip..."
unzip -o data.zip

echo "Preparing dataset directory..."
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

echo "Moving nuscenes info files (safe)..."

NUSC_DIR="${DATASET_DST}/nuscenes"

mkdir -p "${NUSC_DIR}"

for f in "${DATA_DIR}"/vad_nuscenes_infos_temporal_*.pkl; do
  if [[ -f "$f" ]]; then
    echo "Moving $(basename "$f") → nuscenes/"
    mv "$f" "${NUSC_DIR}/"
  else
    echo "Skipping (not found)"
  fi
done

echo "Final nuscenes directory:"
ls -lh "${NUSC_DIR}" || true

echo "Done."