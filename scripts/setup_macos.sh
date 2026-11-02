#!/usr/bin/env bash
# Native macOS environment. DEVICE=cpu permits CPU-only Macs and interpreters.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv-mac}"
DEVICE="${DEVICE:-mps}"
if [[ "${1:-}" != "" && "${1:-}" != "--skip-smoke" ]]; then
  echo "Usage: DEVICE=mps|cpu PYTHON=python3 bash scripts/setup_macos.sh [--skip-smoke]" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup requires macOS. Use scripts/setup_cpu.ps1 on Windows." >&2
  exit 1
fi
if [[ "${DEVICE}" != "mps" && "${DEVICE}" != "cpu" ]]; then
  echo "DEVICE must be mps or cpu." >&2
  exit 1
fi
"${PYTHON}" - "${DEVICE}" <<'PY'
import platform
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required. Python 3.11/3.12 is recommended.")
if sys.argv[1] == "mps" and platform.machine() != "arm64":
    raise SystemExit("MPS setup requires native arm64 Python on Apple silicon. Disable Rosetta or use DEVICE=cpu.")
PY
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON}" -m venv "${VENV_PATH}"
fi
VENV_PYTHON="${VENV_PATH}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install torch -r "${ROOT}/requirements.txt" -r "${ROOT}/requirements-mt5.txt"
cd "${ROOT}"
"${VENV_PYTHON}" - "${DEVICE}" <<'PY'
import platform
import sys
import torch
from utils.device_utils import resolve_task_device
device = resolve_task_device(sys.argv[1])
if device.type == "mps" and platform.machine() != "arm64":
    raise SystemExit("The existing virtual environment uses Rosetta. Select a new VENV_PATH with native arm64 Python.")
if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
    raise SystemExit("This PyTorch build lacks Gloo. Install a macOS build with distributed/Gloo support.")
print("PyTorch:", torch.__version__, "Python:", platform.machine(), "compute:", device, "Gloo: available")
PY
if [[ "${1:-}" != "--skip-smoke" ]]; then
  "${VENV_PYTHON}" "${ROOT}/scripts/run_macos_smoke.py" --device "${DEVICE}"
fi
echo "macOS environment ready: ${VENV_PATH}"
echo "Activate with: source \"${VENV_PATH}/bin/activate\""
