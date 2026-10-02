#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="215abfb217dbac7d5f1273282331b9b1866c0479"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="${1:-$SCRIPT_DIR/openpi}"

if [[ ! -d "$OPENPI_DIR/.git" ]]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi

git -C "$OPENPI_DIR" fetch origin "$PINNED_COMMIT"
git -C "$OPENPI_DIR" checkout --detach "$PINNED_COMMIT"
git -C "$OPENPI_DIR" submodule update --init --recursive

install -D \
  "$SCRIPT_DIR/openpi_overlay/src/openpi/models_pytorch/pi0_ecl_pytorch.py" \
  "$OPENPI_DIR/src/openpi/models_pytorch/pi0_ecl_pytorch.py"

if ! grep -q 'name="pi05_libero_ecl"' "$OPENPI_DIR/src/openpi/training/config.py"; then
  git -C "$OPENPI_DIR" apply --check "$SCRIPT_DIR/openpi_overlay.patch"
  git -C "$OPENPI_DIR" apply "$SCRIPT_DIR/openpi_overlay.patch"
fi

echo "OpenPI ECL overlay installed at: $OPENPI_DIR"
echo "Pinned OpenPI commit: $PINNED_COMMIT"

