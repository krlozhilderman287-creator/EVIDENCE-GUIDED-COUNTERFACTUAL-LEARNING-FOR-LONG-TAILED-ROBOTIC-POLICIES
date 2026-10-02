#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/experiments/ecl${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/.libero}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS:-1}" \
  experiments/ecl/train.py --config configs/ecl_libero_core.json "$@"
