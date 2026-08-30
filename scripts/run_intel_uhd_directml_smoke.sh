#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DIRECTML_ID="${DIRECTML_ID:-0}"
EXPECTED_DIRECTML_NAME="${EXPECTED_DIRECTML_NAME:-Intel}"

cd "${REPO_ROOT}"

DIRECTML_ID="${DIRECTML_ID}" EXPECTED_DIRECTML_NAME="${EXPECTED_DIRECTML_NAME}" \
"${PYTHON_BIN}" - <<'PY'
import os

import torch
import torch_directml

index = int(os.environ['DIRECTML_ID'])
expected_name = os.environ['EXPECTED_DIRECTML_NAME']
count = torch_directml.device_count()
if index < 0 or index >= count:
    raise SystemExit(
        'DIRECTML_ID {} is invalid; DirectML reports {} device(s)'.format(index, count)
    )

name = torch_directml.device_name(index)
if expected_name.lower() not in name.lower():
    raise SystemExit(
        "DirectML adapter {} is '{}', not the expected '{}'. Refusing to test the wrong GPU.".format(
            index, name, expected_name
        )
    )

device = torch_directml.device(index)
x = torch.randn((32, 32), device=device, requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
loss.item()
if x.grad is None:
    raise SystemExit('DirectML backward preflight did not create a gradient')

print('PyTorch:', torch.__version__)
print('Selected index:', index)
print('Selected device:', name)
print('DirectML preflight loss:', loss.item())
PY

"${PYTHON_BIN}" -u dist_runner.py \
  --device directml --directml-id "${DIRECTML_ID}" \
  --world-size 1 --pipeline-group-size 1 --data-group-size 1 --rank 0 \
  --synthetic-data true --synthetic-samples 8 --synthetic-vocab-size 512 \
  --seq-length 32 --embedding-dim 64 --num-layers 1 --num-heads 4 \
  --batch-size 2 --micro-batch-size 1 --num-iters 2 \
  --skip-comm-probe true --use-offload false --profiling no-profiling \
  --metrics-dir ./logs/smoke

echo "DT-FM Intel UHD DirectML smoke test passed."
