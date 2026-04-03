#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"
DATA_DIR="${SSR_DIR}/data"
DATA_ROOT="${REPO_ROOT}/data"
DATASET_DST="${DATA_ROOT}/dataset"
PY39_ENV="/kaggle/working/py39_ssr"

echo "Preparing SSR directory..."
mkdir -p "${SSR_DIR}"

echo "Ensuring SSR data symlink exists..."
mkdir -p "${DATA_ROOT}"

if [[ -L "${DATA_DIR}" ]]; then
  echo "Symlink already exists: ${DATA_DIR} -> $(readlink "${DATA_DIR}")"
elif [[ -e "${DATA_DIR}" ]]; then
  echo "Existing non-symlink path found at ${DATA_DIR}; leaving it untouched for safety."
else
  ln -s "${DATA_ROOT}" "${DATA_DIR}"
  echo "Linked ${DATA_DIR} -> ${DATA_ROOT}"
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

  for f in "${DATA_ROOT}"/vad_nuscenes_infos_temporal_*.pkl; do
    if [[ -f "$f" ]]; then
      echo "Moving $(basename "$f") -> nuscenes/"
      mv "$f" "${NUSC_DIR}/"
    fi
  done
fi

echo "Final symlink check:"
ls -ld "${DATA_DIR}" || true

echo "Final nuscenes directory:"
ls -lh "${DATA_DIR}/dataset/nuscenes" || true

echo "Done."