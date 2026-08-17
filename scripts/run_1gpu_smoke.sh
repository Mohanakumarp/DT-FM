#!/usr/bin/env bash
# Single-process, single-GPU GPipe smoke test (world-size=1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${DTFM_CONDA_ENV:-dtfm}"
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

PORT="${PORT:-9000}"
ITERS="${ITERS:-1}"

python dist_runner.py \
  --use-cuda true \
  --cuda-id 0 \
  --cuda-num 1 \
  --dist-backend cupy_nccl \
  --dist-url "tcp://127.0.0.1:${PORT}" \
  --world-size 1 \
  --pipeline-group-size 1 \
  --data-group-size 1 \
  --rank 0 \
  --seq-length 64 \
  --embedding-dim 128 \
  --num-layers 2 \
  --num-heads 4 \
  --batch-size 1 \
  --micro-batch-size 1 \
  --num-iters "${ITERS}" \
  --train-data ./task_datasets/data/QQP/train.tsv \
  --valid-data ./task_datasets/data/QQP/dev.tsv \
  --test-data ./task_datasets/data/QQP/test.tsv \
  --vocab-file ./task_datasets/data/bert-large-cased-vocab.txt \
  --pp-mode gpipe \
  --dp-mode allreduce \
  --use-offload false \
  --profiling no-profiling
