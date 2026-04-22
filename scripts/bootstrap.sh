#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .

echo "Bootstrapped .venv and installed the project."
