#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATA_DIR="${SSR_DIR}/data"
OUTSIDE_DATA_DIR="${REPO_ROOT}/data"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"

echo "Preparing SSR directory..."
mkdir -p "${SSR_DIR}"

echo "Resolving data directory..."
if [[ -L "${DATA_DIR}" ]]; then
  echo "Symlink already exists: ${DATA_DIR} -> $(readlink "${DATA_DIR}")"
  REAL_DATA_ROOT="$(readlink -f "${DATA_DIR}")"
elif [[ -e "${DATA_DIR}" ]]; then
  echo "Existing real directory found at ${DATA_DIR}; using it directly."
  REAL_DATA_ROOT="${DATA_DIR}"
else
  mkdir -p "${OUTSIDE_DATA_DIR}"
  ln -s "${OUTSIDE_DATA_DIR}" "${DATA_DIR}"
  echo "Linked ${DATA_DIR} -> ${OUTSIDE_DATA_DIR}"
  REAL_DATA_ROOT="${OUTSIDE_DATA_DIR}"
fi

DATASET_DST="${REAL_DATA_ROOT}/dataset"
NUSC_DIR="${DATASET_DST}/nuscenes"

echo "Using data root: ${REAL_DATA_ROOT}"

echo "Checking whether prepared data already exists..."
if [[ -f "${NUSC_DIR}/vad_nuscenes_infos_temporal_train.pkl" ]]; then
  echo "Prepared data already exists, skipping extraction and dataset copy."
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

  echo "Moving nuscenes info files..."
  mkdir -p "${NUSC_DIR}"

  for f in "${REAL_DATA_ROOT}"/vad_nuscenes_infos_temporal_*.pkl; do
    if [[ -f "$f" ]]; then
      echo "Moving $(basename "$f") -> nuscenes/"
      mv "$f" "${NUSC_DIR}/"
    fi
  done
fi

echo "Final path check:"
ls -ld "${DATA_DIR}" || true
echo "Resolved data root:"
readlink -f "${DATA_DIR}" || echo "${REAL_DATA_ROOT}"

echo "Final nuscenes directory:"
ls -lh "${NUSC_DIR}" || true

echo "Done."