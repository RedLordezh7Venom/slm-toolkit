#!/usr/bin/env bash
# Run slm_converter.py using the llama.cpp venv if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/llama.cpp/.venv/bin/python3"

if [ -f "$VENV" ]; then
    exec "$VENV" "$SCRIPT_DIR/converter.py" "$@"
else
    exec python3 "$SCRIPT_DIR/converter.py" "$@"
fi
