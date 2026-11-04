#!/usr/bin/env bash
# macOS/Linux: create .venv, install the pinned requirements, run the pipeline.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python main.py
