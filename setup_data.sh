#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"

# Real data lives here
DATA_DIR="${SSR_DIR}/data"

# This becomes a symlink to DATA_DIR
DATA_ROOT="${REPO_ROOT}/data"

DATASET_DST="${DATA_DIR}/dataset"
PY39_ENV="/kaggle/working/py39_ssr"

echo "Preparing SSR directory..."
mkdir -p "${SSR_DIR}"
mkdir -p "${DATA_DIR}"

echo "Ensuring ${DATA_ROOT} points to ${DATA_DIR}..."

if [[ -L "${DATA_ROOT}" ]]; then
  echo "Symlink already exists: ${DATA_ROOT} -> $(readlink "${DATA_ROOT}")"
elif [[ -e "${DATA_ROOT}" ]]; then
  echo "Existing non-symlink path found at ${DATA_ROOT}; leaving it untouched for safety."
else
  ln -s "${DATA_DIR}" "${DATA_ROOT}"
  echo "Linked ${DATA_ROOT} -> ${DATA_DIR}"
fi

echo "Checking whether prepared data already exists..."
if [[ -d "${DATA_DIR}/dataset/nuscenes" ]] && [[ -n "$(ls -A "${DATA_DIR}/dataset/nuscenes" 2>/dev/null || true)" ]]; then
  echo "Prepared data already found at ${DATA_DIR}, skipping extraction and dataset copy."
else
  echo "Reconstructing data.zip..."
  cd "${SSR_DIR}"
  rm -f data.zip
  cat data.zip.part-a* > data.zip

  echo "Unzipping data.zip..."
  unzip -o data.zip

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
      echo "Moving $(basename "$f") -> nuscenes/"
      mv "$f" "${NUSC_DIR}/"
    fi
  done
fi

echo "Final real data dir check:"
ls -ld "${DATA_DIR}" || true

echo "Final symlink check:"
ls -ld "${DATA_ROOT}" || true

echo "Resolved /kaggle/working/QuantizedSSR/data:"
readlink -f "${DATA_ROOT}" || true

echo "Final nuscenes directory:"
ls -lh "${DATA_DIR}/dataset/nuscenes" || true

echo "Done."