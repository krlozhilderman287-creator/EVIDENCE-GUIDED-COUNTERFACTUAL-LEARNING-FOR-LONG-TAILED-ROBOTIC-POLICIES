#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$SCRIPT_DIR/openpi}"
CHECKPOINT_DIR="${1:?usage: evaluate.sh CHECKPOINT_DIR [LIBERO client arguments...]}"
shift

cd "$OPENPI_DIR"
echo "Terminal 1:"
echo "  uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi05_libero_ecl --policy.dir '$CHECKPOINT_DIR'"
echo "Terminal 2:"
printf "  MUJOCO_GL=egl uv run examples/libero/main.py"
printf " %q" "$@"
printf "\n"

