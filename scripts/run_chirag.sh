#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Create a Python 3.11 environment and install .[training,dev] first; see README.md.'
  exit 1
fi
export OLLAMA_HOST=127.0.0.1:11434
export HG_CLASSIFIER_DEVICE="${HG_CLASSIFIER_DEVICE:-cpu}"
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
