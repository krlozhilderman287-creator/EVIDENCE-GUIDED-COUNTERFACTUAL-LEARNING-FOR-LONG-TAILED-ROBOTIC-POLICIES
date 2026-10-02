#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT/tensorflow_datasets}"
export LIBERO_CORE_FULL_HDF5="${LIBERO_CORE_FULL_HDF5:-$ROOT/dataset_all/libero_core_full_no_noops}"
export LIBERO_CORE_LT_HDF5="${LIBERO_CORE_LT_HDF5:-$ROOT/dataset_all/libero_core_lt_no_noops}"
export CUDA_VISIBLE_DEVICES=""
for variant in full lt; do
  (cd "$ROOT/rlds_dataset_builder/libero_core_$variant" && tfds build --data_dir "$TFDS_DATA_DIR")
done
