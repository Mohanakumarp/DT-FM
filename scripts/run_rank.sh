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
DEVICE="${DEVICE:-auto}"

if [[ -z "${PYTHON:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${DTFM_CONDA_ENV:-dtfm}"
fi
PYTHON="${PYTHON:-python}"
if [[ "${DEVICE}" == "cuda" || "${DEVICE}" == "auto" ]]; then
  # The legacy CUDA path needs its CuPy/NCCL library setup. Loading those
  # libraries in ROCm, XPU, or DirectML environments can initialize the wrong
  # accelerator runtime.
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.sh"
else
  export DTFM_ROOT="${ROOT}"
fi
cd "${ROOT}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

RANK="${RANK:?set RANK=0, 1, or 2}"
WORLD_SIZE="${WORLD_SIZE:-3}"
PP_SIZE="${PP_SIZE:-${WORLD_SIZE}}"
DP_SIZE="${DP_SIZE:-1}"
PORT="${PORT:-9000}"
LOG_PORT="${LOG_PORT:-9100}"
ITERS="${ITERS:-70}"
EPOCHS="${EPOCHS:-0}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"
PROFILE="${PROFILE:-tiny}"
TENSOR_COMM="${TENSOR_COMM:-gloo}"
EXPECTED_DIRECTML_NAME="${EXPECTED_DIRECTML_NAME:-Intel}"
SYNTHETIC_DATA="${SYNTHETIC_DATA:-false}"
SYNTHETIC_SAMPLES="${SYNTHETIC_SAMPLES:-32}"
SYNTHETIC_VOCAB_SIZE="${SYNTHETIC_VOCAB_SIZE:-2048}"
SEQ="${SEQ:-64}"
EMBED="${EMBED:-128}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
SCHEDULE_ARGS=()
if [[ -n "${DYNAMIC_TOTAL_LAYERS:-}" ]]; then
  if [[ -n "${STAGE_LAYERS:-}" ]]; then
    echo "DYNAMIC_TOTAL_LAYERS cannot be combined with STAGE_LAYERS." >&2
    exit 1
  fi
  SCHEDULE_ARGS+=(--dynamic-total-layers "${DYNAMIC_TOTAL_LAYERS}"
                 --scheduler-memory-fraction "${SCHEDULER_MEMORY_FRACTION:-0.7}"
                 --scheduler-reserve-mb "${SCHEDULER_RESERVE_MB:-256}"
                 --scheduler-warmup "${SCHEDULER_WARMUP:-1}"
                 --scheduler-repeats "${SCHEDULER_REPEATS:-3}")
  SCHEDULE_ARGS+=(--rebalance-every "${REBALANCE_EVERY:-0}"
                 --rebalance-min-improvement "${REBALANCE_MIN_IMPROVEMENT:-0.1}")
  if [[ -n "${SCHEDULER_HOST_ID:-}" ]]; then
    SCHEDULE_ARGS+=(--scheduler-host-id "${SCHEDULER_HOST_ID}")
  fi
fi
if [[ -n "${STAGE_LAYERS:-}" ]]; then
  if [[ "${PROFILE}" =~ ^bert ]]; then
    echo "STAGE_LAYERS cannot be combined with PROFILE=${PROFILE}; use the generated scheduled command." >&2
    exit 1
  fi
  SCHEDULE_ARGS+=(--stage-layers "${STAGE_LAYERS}")
fi
if [[ -n "${SCHEDULE_VOCAB_SIZE:-}" ]]; then
  SCHEDULE_ARGS+=(--schedule-vocab-size "${SCHEDULE_VOCAB_SIZE}")
fi

# BERT profiles for QQP fine-tuning:
# 1. bert (base ~110M params): 12 layers (4+4+4 across 3 ranks), hidden=768, heads=12
# 2. bert-500m (~510M params): 24 layers (8+8+8 across 3 ranks), hidden=1280, heads=16
#    Optimized for 3x 6GB RTX 3050 GPUs (~3.2 GB VRAM per rank, fits cleanly).
# 3. bert-large (~340M params): 24 layers (8+8+8 across 3 ranks), hidden=1024, heads=16
if [[ "${PROFILE}" == "bert" || "${PROFILE}" == "bert-base" ]]; then
  SEQ=128
  EMBED=768
  LAYERS=4
  HEADS=12
  if [[ "${EPOCHS}" -eq 0 ]]; then EPOCHS=2; fi
  if [[ "${STEPS_PER_EPOCH}" -eq 0 ]]; then STEPS_PER_EPOCH=50; fi
  ITERS=$((EPOCHS * STEPS_PER_EPOCH))
elif [[ "${PROFILE}" == "bert-500m" ]]; then
  SEQ=128
  EMBED=1280
  LAYERS=8
  HEADS=16
  if [[ "${EPOCHS}" -eq 0 ]]; then EPOCHS=2; fi
  if [[ "${STEPS_PER_EPOCH}" -eq 0 ]]; then STEPS_PER_EPOCH=50; fi
  ITERS=$((EPOCHS * STEPS_PER_EPOCH))
elif [[ "${PROFILE}" == "bert-large" ]]; then
  SEQ=128
  EMBED=1024
  LAYERS=8
  HEADS=16
  if [[ "${EPOCHS}" -eq 0 ]]; then EPOCHS=2; fi
  if [[ "${STEPS_PER_EPOCH}" -eq 0 ]]; then STEPS_PER_EPOCH=50; fi
  ITERS=$((EPOCHS * STEPS_PER_EPOCH))
fi

if [[ -z "${MASTER_IP:-}" ]]; then
  if command -v tailscale >/dev/null 2>&1; then
    MASTER_IP="$(tailscale ip -4 | head -n 1 || true)"
  fi
fi
if [[ -z "${MASTER_IP:-}" ]]; then
  echo "Set MASTER_IP to the rank-0 Tailscale IPv4." >&2
  exit 1
fi

# env.sh defaults GLOO to lo for local smokes. Multi-laptop runs must use
# tailscale0 or the peer cannot connect to MASTER_IP.
if [[ "${DTFM_LOCAL:-0}" == "1" ]]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) unset GLOO_SOCKET_IFNAME ;;
    *) export GLOO_SOCKET_IFNAME=lo ;;
  esac
elif [[ -d /sys/class/net/tailscale0 ]]; then
  export GLOO_SOCKET_IFNAME=tailscale0
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

# MLflow tracking
MLFLOW_ARGS=()
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  DEFAULT_EXPERIMENT="DT-FM training"
  if [[ "${PROFILE}" =~ ^bert ]]; then
    DEFAULT_EXPERIMENT="DT-FM $(echo "${PROFILE}" | tr '[:lower:]' '[:upper:]')"
  fi
  MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-${DEFAULT_EXPERIMENT}}"
  MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-${PROFILE}-rank-${RANK}}"
  MLFLOW_ARGS+=(--mlflow-tracking-uri "${MLFLOW_TRACKING_URI}"
                --mlflow-experiment "${MLFLOW_EXPERIMENT}"
                --mlflow-run-name "${MLFLOW_RUN_NAME}")
  if [[ -n "${MLFLOW_GROUP:-}" ]]; then
    MLFLOW_ARGS+=(--mlflow-group "${MLFLOW_GROUP}")
  fi
fi

if [[ "${RANK}" == "0" ]]; then
  "${PYTHON}" -u "${SCRIPT_DIR}/log_hub.py" --port "${LOG_PORT}" --out "${LOG_DIR}/all_ranks.log" &
  HUB_PID=$!
  sleep 0.4
  echo "[run_rank] log hub on ${MASTER_IP}:${LOG_PORT}  (live: tail -F ${LOG_DIR}/all_ranks.log)"
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    echo "[run_rank] MLflow tracking: ${MLFLOW_TRACKING_URI} (experiment: ${MLFLOW_EXPERIMENT})"
  fi
  if [[ -n "${DYNAMIC_TOTAL_LAYERS:-}${STAGE_LAYERS:-}" ]]; then
    echo "[run_rank] start each peer with its emitted manifest command and matching model/batch settings."
  else
    echo "[run_rank] start the other laptops with:"
    for ((r=1; r<WORLD_SIZE; r++)); do
      MLFLOW_PEER_ENV=""
      if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
        MLFLOW_PEER_ENV="MLFLOW_TRACKING_URI='${MLFLOW_TRACKING_URI}' MLFLOW_EXPERIMENT='${MLFLOW_EXPERIMENT}' "
        if [[ -n "${MLFLOW_GROUP:-}" ]]; then
          MLFLOW_PEER_ENV+="MLFLOW_GROUP='${MLFLOW_GROUP}' "
        fi
      fi
      echo "  ${MLFLOW_PEER_ENV}DEVICE=<device-for-rank-${r}> RANK=${r} MASTER_IP=${MASTER_IP} WORLD_SIZE=${WORLD_SIZE} PROFILE=${PROFILE} EPOCHS=${EPOCHS} STEPS_PER_EPOCH=${STEPS_PER_EPOCH} ITERS=${ITERS} BATCH=${BATCH:-8} MICRO=${MICRO:-2} SKIP_PROBE=${SKIP_PROBE:-true} bash scripts/run_rank.sh"
    done
  fi
fi

echo "[run_rank] rank=${RANK}/${WORLD_SIZE} device=${DEVICE} master=${MASTER_IP}:${PORT} profile=${PROFILE} synthetic=${SYNTHETIC_DATA} epochs=${EPOCHS} spe=${STEPS_PER_EPOCH} iters=${ITERS} iface=${GLOO_SOCKET_IFNAME:-auto}"

"${PYTHON}" -u dist_runner.py \
  "${SCHEDULE_ARGS[@]}" \
  "${MLFLOW_ARGS[@]}" \
  --device "${DEVICE}" \
  --cuda-id "${CUDA_ID:-0}" \
  --directml-id "${DIRECTML_ID:-0}" \
  --directml-expected-name "${EXPECTED_DIRECTML_NAME}" \
  --cuda-num 1 \
  --dist-backend gloo \
  --tensor-comm "${TENSOR_COMM}" \
  --dist-url "tcp://${MASTER_IP}:${PORT}" \
  --world-size "${WORLD_SIZE}" \
  --pipeline-group-size "${PP_SIZE}" \
  --data-group-size "${DP_SIZE}" \
  --rank "${RANK}" \
  --synthetic-data "${SYNTHETIC_DATA}" \
  --synthetic-samples "${SYNTHETIC_SAMPLES}" \
  --synthetic-vocab-size "${SYNTHETIC_VOCAB_SIZE}" \
  --seq-length "${SEQ}" \
  --embedding-dim "${EMBED}" \
  --num-layers "${LAYERS}" \
  --num-heads "${HEADS}" \
  --batch-size "${BATCH:-8}" \
  --micro-batch-size "${MICRO:-2}" \
  --num-iters "${ITERS}" \
  --num-epochs "${EPOCHS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --metrics-dir "${LOG_DIR}" \
  --skip-comm-probe "${SKIP_PROBE:-true}" \
  --train-data ./task_datasets/data/QQP/train.tsv \
  --valid-data ./task_datasets/data/QQP/dev.tsv \
  --test-data ./task_datasets/data/QQP/test.tsv \
  --vocab-file ./task_datasets/data/bert-large-cased-vocab.txt \
  --pp-mode gpipe \
  --dp-mode allreduce \
  --use-offload false \
  --profiling no-profiling \
  2>&1 | "${PYTHON}" -u "${SCRIPT_DIR}/log_tee.py" \
      --rank "${RANK}" \
      --hub "${MASTER_IP}:${LOG_PORT}" \
      --file "${LOG_DIR}/rank${RANK}.log"
