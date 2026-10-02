#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Exercise masked forward + counterfactual backward from the first step.
NUM_GPUS=1 bash "$ROOT/scripts/ecl/train.sh" \
  --run-root runs/smoke --per-device-batch-size 1 --global-batch-size 1 \
  --max-steps 2 --warmup-epochs 0 --save-interval 2 "$@"
