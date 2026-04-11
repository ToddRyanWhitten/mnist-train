#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# activate venv
source "$VENV_DIR/bin/activate"

# Install requirements if requirements.txt exists and packages are outdated/missing
if [ -f "$REQUIREMENTS" ]; then
    echo "Checking pip requirements..."
    pip install -q -r "$REQUIREMENTS"
fi

echo "Environment ready"
