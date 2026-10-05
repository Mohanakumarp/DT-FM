#!/usr/bin/env bash
set -euo pipefail

case "${1:-smoke}" in
  mt5)
    shift
    exec python -u /app/scripts/train_mt5_pipeline.py --device "${DTFM_DEVICE:-cpu}" "$@"
    ;;
  smoke)
    if [[ $# -gt 0 ]]; then shift; fi
    ranks="${DTFM_WORLD_SIZE:-1}"
    if [[ ! "${ranks}" =~ ^[1-9][0-9]*$ ]]; then
      echo 'DTFM_WORLD_SIZE must be a positive integer.' >&2
      exit 2
    fi
    exec python -u /app/dist_runner.py \
      --device "${DTFM_DEVICE:-cpu}" \
      --world-size "${ranks}" --pipeline-group-size "${ranks}" \
      --data-group-size 1 --rank 0 --spawn-local-ranks true \
      --tensor-comm gloo --dist-backend gloo \
      --dist-url tcp://127.0.0.1:9000 --dist-timeout-seconds 60 \
      --synthetic-data true --synthetic-samples 8 --synthetic-vocab-size 512 \
      --seq-length 32 --embedding-dim 64 --num-layers 1 --num-heads 4 \
      --batch-size 2 --micro-batch-size 1 --num-iters 2 \
      --use-offload false --profiling no-profiling \
      --metrics-dir ./logs/smoke "$@"
    ;;
  train)
    shift
    exec python -u /app/dist_runner.py "$@"
    ;;
  test)
    shift
    cd /app
    exec python -m unittest discover -s tests -v "$@"
    ;;
  -*)
    exec python -u /app/dist_runner.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
