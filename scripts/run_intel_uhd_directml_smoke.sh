#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DIRECTML_ID="${DIRECTML_ID:-0}"
EXPECTED_DIRECTML_NAME="${EXPECTED_DIRECTML_NAME:-Intel}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -u dist_runner.py \
  --device directml --directml-id "${DIRECTML_ID}" \
  --directml-expected-name "${EXPECTED_DIRECTML_NAME}" \
  --world-size 1 --pipeline-group-size 1 --data-group-size 1 --rank 0 \
  --synthetic-data true --synthetic-samples 8 --synthetic-vocab-size 512 \
  --seq-length 32 --embedding-dim 64 --num-layers 1 --num-heads 4 \
  --batch-size 2 --micro-batch-size 1 --num-iters 2 \
  --skip-comm-probe true --use-offload false --profiling no-profiling \
  --metrics-dir ./logs/smoke

echo "DT-FM Intel UHD DirectML smoke test passed."
