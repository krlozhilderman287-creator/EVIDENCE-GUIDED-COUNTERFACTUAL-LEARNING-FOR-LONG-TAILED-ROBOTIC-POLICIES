#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$SCRIPT_DIR/openpi}"
ECL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_JAX_CHECKPOINT="${BASE_JAX_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}"
BASE_TORCH_CHECKPOINT="${BASE_TORCH_CHECKPOINT:-$OPENPI_DIR/checkpoints/pi05_base_pytorch}"
RUN_NAME="${RUN_NAME:-ecl_core_lt_s7}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-7}"
CONVERT_DATA="${CONVERT_DATA:-1}"

cd "$OPENPI_DIR"
uv sync
uv pip install tensorflow tensorflow-datasets
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

if [[ "$CONVERT_DATA" == "1" ]]; then
  uv run "$SCRIPT_DIR/convert_core_lt_to_lerobot.py" \
    --data_dir "$ECL_ROOT/tensorflow_datasets"
fi

uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$BASE_JAX_CHECKPOINT" \
  --config_name pi05_libero \
  --output_path "$BASE_TORCH_CHECKPOINT"

uv run scripts/compute_norm_stats.py --config-name pi05_libero_ecl

uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  scripts/train_pytorch.py pi05_libero_ecl \
  --exp_name "$RUN_NAME" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --pytorch_weight_path "$BASE_TORCH_CHECKPOINT"
