#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/QuantizedSSR"
SSR_DIR="${REPO_ROOT}/ssr"
DATASET_SRC="/kaggle/input/datasets/vuthanhphong/ssr-dataset/dataset"

# real storage
REAL_DATA_DIR="${SSR_DIR}/data"

# path expected by code
LINK_DATA_DIR="${REPO_ROOT}/data"

DATASET_DST="${REAL_DATA_DIR}/dataset"
NUSC_DIR="${DATASET_DST}/nuscenes"

echo "Preparing directories..."
mkdir -p "${SSR_DIR}"
mkdir -p "${REAL_DATA_DIR}"

echo "Ensuring ${LINK_DATA_DIR} points to ${REAL_DATA_DIR} ..."
if [[ -L "${LINK_DATA_DIR}" ]]; then
  echo "Existing symlink: ${LINK_DATA_DIR} -> $(readlink "${LINK_DATA_DIR}")"
  rm -f "${LINK_DATA_DIR}"
elif [[ -e "${LINK_DATA_DIR}" ]]; then
  echo "ERROR: ${LINK_DATA_DIR} exists as a real file/directory."
  echo "Move it away first:"
  echo "  mv ${LINK_DATA_DIR} ${LINK_DATA_DIR}.bak"
  exit 1
fi

ln -s "${REAL_DATA_DIR}" "${LINK_DATA_DIR}"
echo "Linked ${LINK_DATA_DIR} -> ${REAL_DATA_DIR}"

echo "Reconstructing data.zip if needed..."
cd "${SSR_DIR}"
if [[ ! -f data.zip ]]; then
  cat data.zip.part-a* > data.zip
fi

echo "Unzipping data.zip..."
unzip -o data.zip

echo "Copying dataset..."
rm -rf "${DATASET_DST}"
mkdir -p "${DATASET_DST}"
cp -r "${DATASET_SRC}/." "${DATASET_DST}/"

echo "Ensuring nuscenes directory exists..."
mkdir -p "${NUSC_DIR}"

echo "Moving nuscenes info PKLs into dataset/nuscenes ..."
find "${REAL_DATA_DIR}" -maxdepth 1 -type f -name 'vad_nuscenes_infos_temporal_*.pkl' -print -exec mv -f {} "${NUSC_DIR}/" \;

echo "Verifying required files..."
test -f "${NUSC_DIR}/vad_nuscenes_infos_temporal_train.pkl"
test -f "${NUSC_DIR}/vad_nuscenes_infos_temporal_val.pkl"

echo "Symlink check:"
ls -ld "${LINK_DATA_DIR}"
echo "Resolved:"
readlink -f "${LINK_DATA_DIR}"

echo "Final nuscenes contents:"
ls -lah "${NUSC_DIR}"

echo "Done."