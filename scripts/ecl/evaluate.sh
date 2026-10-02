#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/.libero}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python experiments/ecl/evaluate.py "$@"
