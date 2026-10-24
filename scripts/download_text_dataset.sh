#!/usr/bin/env bash
# Download and prepare WikiText-103 dataset for DT-FM GPT-2 training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${SCRIPT_DIR}/download_text_dataset.py"
