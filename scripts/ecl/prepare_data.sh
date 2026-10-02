#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/.libero}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
for suite in libero_spatial libero_object libero_goal; do
  python scripts/dataset/parallel_libero_dataset_regenerator.py \
    --num-gpus "${NUM_GPUS:-1}" --max-processes "${MAX_PROCESSES:-1}" \
    --libero-task-suite "$suite" --libero-raw-data-dir "$ROOT/libero_raw/$suite" \
    --libero-target-dir "$ROOT/dataset_all/${suite}_no_noops"
done
python scripts/dataset/create_libero_core_full.py --dataset_root "$ROOT/dataset_all"
python scripts/dataset/create_libero_core_lt.py \
  --source_dir "$ROOT/dataset_all/libero_core_full_no_noops" --seed 100
python scripts/ecl/check_dataset.py --variant full
python scripts/ecl/check_dataset.py --variant lt
