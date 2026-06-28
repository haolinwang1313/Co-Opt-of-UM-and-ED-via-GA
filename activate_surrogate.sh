#!/usr/bin/env bash
# Helper to activate the shared GeoEnergy virtualenv from Paper03.
set -euo pipefail
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_PATH="$SCRIPT_DIR/.venv_geo"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
echo "Activating surrogate environment from $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
