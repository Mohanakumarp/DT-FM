#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RANK0_PYTHON="${RANK0_PYTHON:-python}"
RANK1_PYTHON="${RANK1_PYTHON:-python}"
RANK0_DEVICE="${RANK0_DEVICE:-cpu}"
RANK1_DEVICE="${RANK1_DEVICE:-directml}"
DIRECTML_ID="${DIRECTML_ID:-0}"
EXPECTED_DIRECTML_NAME="${EXPECTED_DIRECTML_NAME:-Intel}"
PORT="${PORT:-9031}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
ITERATIONS="${ITERATIONS:-1}"

cd "${REPO_ROOT}"
mkdir -p logs/smoke

COMMON_ARGS=(
  -u dist_runner.py
  --tensor-comm gloo --dist-backend gloo
  --dist-url "tcp://127.0.0.1:${PORT}"
  --world-size 2 --pipeline-group-size 2 --data-group-size 1
  --directml-id "${DIRECTML_ID}"
  --directml-expected-name "${EXPECTED_DIRECTML_NAME}"
  --synthetic-data true --synthetic-samples 8 --synthetic-vocab-size 512
  --seq-length 32 --embedding-dim 64 --num-layers 1 --num-heads 4
  --batch-size 2 --micro-batch-size 1 --num-iters "${ITERATIONS}"
  --metrics-dir ./logs/smoke --skip-comm-probe true
  --use-offload false --profiling no-profiling
)

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

timeout "${TIMEOUT_SECONDS}" "${RANK0_PYTHON}" "${COMMON_ARGS[@]}" \
  --device "${RANK0_DEVICE}" --rank 0 \
  >logs/smoke/mixed_rank0.log 2>&1 &
PIDS+=("$!")

timeout "${TIMEOUT_SECONDS}" "${RANK1_PYTHON}" "${COMMON_ARGS[@]}" \
  --device "${RANK1_DEVICE}" --rank 1 \
  >logs/smoke/mixed_rank1.log 2>&1 &
PIDS+=("$!")

set +e
wait "${PIDS[0]}"
RANK0_EXIT=$?
wait "${PIDS[1]}"
RANK1_EXIT=$?
set -e

cat logs/smoke/mixed_rank0.log
cat logs/smoke/mixed_rank1.log

if [[ "${RANK0_EXIT}" -ne 0 || "${RANK1_EXIT}" -ne 0 ]]; then
  echo "Mixed-device smoke failed: rank0=${RANK0_EXIT} rank1=${RANK1_EXIT}" >&2
  exit 1
fi

echo "Mixed-device CPU-staged Gloo smoke test passed."
