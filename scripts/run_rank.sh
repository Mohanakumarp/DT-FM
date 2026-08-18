#!/usr/bin/env bash
# Launch one DT-FM rank and forward its logs to rank 0.
#
#   # laptop A (master)
#   RANK=0 WORLD_SIZE=3 bash scripts/run_rank.sh
#
#   # laptops B and C
#   RANK=1 MASTER_IP=100.x.x.x WORLD_SIZE=3 bash scripts/run_rank.sh
#   RANK=2 MASTER_IP=100.x.x.x WORLD_SIZE=3 bash scripts/run_rank.sh
#
# Rank 0 starts a log hub on LOG_PORT (default 9100). All ranks also write
# logs/rankN.log locally. On rank 0, tail logs/all_ranks.log to see everyone.
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
mkdir -p logs

RANK="${RANK:?set RANK=0, 1, or 2}"
WORLD_SIZE="${WORLD_SIZE:-3}"
PP_SIZE="${PP_SIZE:-${WORLD_SIZE}}"
DP_SIZE="${DP_SIZE:-1}"
PORT="${PORT:-9000}"
LOG_PORT="${LOG_PORT:-9100}"
ITERS="${ITERS:-70}"
TENSOR_COMM="${TENSOR_COMM:-gloo}"
SEQ="${SEQ:-64}"
EMBED="${EMBED:-128}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"

if [[ -z "${MASTER_IP:-}" ]]; then
  if command -v tailscale >/dev/null 2>&1; then
    MASTER_IP="$(tailscale ip -4 | head -n 1 || true)"
  fi
fi
if [[ -z "${MASTER_IP:-}" ]]; then
  echo "Set MASTER_IP to the rank-0 Tailscale IPv4." >&2
  exit 1
fi

if [[ -d /sys/class/net/tailscale0 ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-tailscale0}"
else
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
fi
export PYTHONUNBUFFERED=1

HUB_PID=""
cleanup() {
  if [[ -n "${HUB_PID}" ]] && kill -0 "${HUB_PID}" 2>/dev/null; then
    kill "${HUB_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${RANK}" == "0" ]]; then
  python -u "${SCRIPT_DIR}/log_hub.py" --port "${LOG_PORT}" --out logs/all_ranks.log &
  HUB_PID=$!
  sleep 0.4
  echo "[run_rank] log hub on ${MASTER_IP}:${LOG_PORT}  (live: tail -F logs/all_ranks.log)"
  echo "[run_rank] start the other laptops with:"
  for ((r=1; r<WORLD_SIZE; r++)); do
    echo "  RANK=${r} MASTER_IP=${MASTER_IP} WORLD_SIZE=${WORLD_SIZE} ITERS=${ITERS} bash scripts/run_rank.sh"
  done
fi

echo "[run_rank] rank=${RANK}/${WORLD_SIZE} master=${MASTER_IP}:${PORT} iters=${ITERS} iface=${GLOO_SOCKET_IFNAME}"

python -u dist_runner.py \
  --use-cuda true \
  --cuda-id "${CUDA_ID:-0}" \
  --cuda-num 1 \
  --dist-backend cupy_nccl \
  --tensor-comm "${TENSOR_COMM}" \
  --dist-url "tcp://${MASTER_IP}:${PORT}" \
  --world-size "${WORLD_SIZE}" \
  --pipeline-group-size "${PP_SIZE}" \
  --data-group-size "${DP_SIZE}" \
  --rank "${RANK}" \
  --seq-length "${SEQ}" \
  --embedding-dim "${EMBED}" \
  --num-layers "${LAYERS}" \
  --num-heads "${HEADS}" \
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
  --profiling no-profiling \
  2>&1 | python -u "${SCRIPT_DIR}/log_tee.py" \
      --rank "${RANK}" \
      --hub "${MASTER_IP}:${LOG_PORT}" \
      --file "logs/rank${RANK}.log"
