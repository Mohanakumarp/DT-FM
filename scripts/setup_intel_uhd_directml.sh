#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        'Intel UHD DirectML setup requires Python 3.11; current interpreter is {}'.format(
            sys.version.split()[0]
        )
    )
print('Python:', sys.version.split()[0])
PY

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install torch-directml
"${PYTHON_BIN}" -m pip install -r requirements.txt

"${PYTHON_BIN}" - <<'PY'
import torch
import torch_directml

count = torch_directml.device_count()
if count < 1:
    raise SystemExit('DirectML did not enumerate any graphics adapters')

print('PyTorch:', torch.__version__)
print('DirectML devices:', count)
for index in range(count):
    print('  [{}] {}'.format(index, torch_directml.device_name(index)))
PY

echo "DirectML environment is ready. Run scripts/run_intel_uhd_directml_smoke.sh with DIRECTML_ID set to the Intel adapter index."
