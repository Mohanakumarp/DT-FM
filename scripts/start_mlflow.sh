#!/usr/bin/env bash
# Start the MLflow tracking server for distributed training.
# Listens on 0.0.0.0 so all Tailscale peers and local browser can access it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PORT="${PORT:-5000}"
HOST="${HOST:-0.0.0.0}"

# Find python interpreter
if [[ -z "${PYTHON:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${DTFM_CONDA_ENV:-dtfm}" 2>/dev/null || true
  fi
fi
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -f "${ROOT}/.venv-cpu/bin/python" ]]; then
    PYTHON="${ROOT}/.venv-cpu/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "Python not found. Set PYTHON=/path/to/python" >&2
    exit 1
  fi
fi

# Determine Tailscale IP for display
TAILSCALE_IP=""
if command -v tailscale >/dev/null 2>&1; then
  TAILSCALE_IP="$(tailscale ip -4 | head -n 1 || true)"
fi

mkdir -p mlartifacts logs

echo "=========================================================="
echo " Starting MLflow Tracking Server"
echo "=========================================================="
echo " Binding: ${HOST}:${PORT}"
echo " Database: sqlite:///${ROOT}/mlflow.db"
echo " Artifacts: ${ROOT}/mlartifacts"
echo ""
echo " Local UI:      http://127.0.0.1:${PORT}"
if [[ -n "${TAILSCALE_IP}" ]]; then
  echo " Tailscale UI:  http://${TAILSCALE_IP}:${PORT}"
  echo " Remote nodes:  export MLFLOW_TRACKING_URI=http://${TAILSCALE_IP}:${PORT}"
fi
echo "=========================================================="

exec "${PYTHON}" -m mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "sqlite:///${ROOT}/mlflow.db" \
  --artifacts-destination "${ROOT}/mlartifacts"
