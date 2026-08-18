#!/usr/bin/env bash
# Show the combined training log.
#
#   bash scripts/show_training.sh            # live-follow logs/all_ranks.log
#   bash scripts/show_training.sh --once     # print what is already on disk
#   bash scripts/show_training.sh logs/rank*.log   # merge local rank files
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${1:-}" == "--once" ]]; then
  if [[ -f logs/all_ranks.log ]]; then
    cat logs/all_ranks.log
  else
    echo "No logs/all_ranks.log yet. Run RANK=0 bash scripts/run_rank.sh first." >&2
    exit 1
  fi
  exit 0
fi

if [[ $# -gt 0 ]]; then
  python -u "${ROOT}/scripts/merge_rank_logs.py" "$@"
  exit 0
fi

mkdir -p logs
touch logs/all_ranks.log
echo "Following logs/all_ranks.log (Ctrl-C to stop)"
exec tail -n +1 -F logs/all_ranks.log
