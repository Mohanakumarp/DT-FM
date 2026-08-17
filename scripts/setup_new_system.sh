#!/usr/bin/env bash
# Set up DT-FM on a new Linux / WSL machine with one NVIDIA GPU.
#
# Usage:
#   bash scripts/setup_new_system.sh
#   bash scripts/setup_new_system.sh --skip-smoke
#   bash scripts/setup_new_system.sh --data-only
#   bash scripts/setup_new_system.sh --check
#
# Tested combination:
#   Python 3.8, PyTorch 1.9.0+cu111, CuPy 12.3.0, NCCL via CuPy,
#   CUDA 11.8 runtime from the classic defaults package `cudatoolkit=11.8`.
#   Do NOT install nvidia::cuda-toolkit (hyphenated). That metapackage does
#   not pin cuda-version and the solver will pull CUDA 12/13 libraries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${DTFM_CONDA_ENV:-dtfm}"
PYTHON_VERSION="3.8"
SKIP_SMOKE=0
SKIP_DATA=0
DATA_ONLY=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Set up the DT-FM conda env, CUDA/NCCL libraries, and QQP data.

Options:
  --env NAME       Conda env name (default: dtfm)
  --skip-smoke     Do not run the 1-GPU training smoke test
  --skip-data      Do not download QQP / vocab
  --data-only      Only download data, then exit
  --check          Only print environment sanity checks
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --data-only) DATA_ONLY=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

load_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    die "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

activate_env() {
  load_conda
  conda activate "${ENV_NAME}"
}

is_wsl() {
  grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null
}

install_nvrtc_compat() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    return 0
  fi
  if [[ -e "${CONDA_PREFIX}/lib/libnvrtc.so.11.1" && ! -e "${CONDA_PREFIX}/lib/libnvrtc.so.11.0" ]]; then
    log "Creating libnvrtc.so.11.0 -> libnvrtc.so.11.1 compatibility symlink"
    ln -sfn libnvrtc.so.11.1 "${CONDA_PREFIX}/lib/libnvrtc.so.11.0"
  fi
}

# CuPy's cuda11x wheel dlopens unmangled CUDA 11 SONAMEs (libcublas.so.11,
# libcudart.so.11.0, libcufft.so.10, libcusparse.so.11, libnvrtc.so.11.2).
# The PyTorch 1.9+cu111 wheel does NOT provide those: it only ships a hashed
# libcudart and statically links the rest. A system toolkit of CUDA 12/13
# also cannot satisfy .so.11. The classic Anaconda package `cudatoolkit=11.8`
# (no hyphen) drops the right files in $CONDA_PREFIX/lib.
ensure_cuda11_runtime() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    die "CONDA_PREFIX is empty; activate the env first"
  fi
  if [[ -e "${CONDA_PREFIX}/lib/libcublas.so.11" && -e "${CONDA_PREFIX}/lib/libcudart.so.11.0" ]]; then
    log "CUDA 11 runtime already present (libcublas.so.11, libcudart.so.11.0)"
    return 0
  fi
  if compgen -G "${CONDA_PREFIX}/lib/libcublas.so.1[23]" > /dev/null \
     || [[ -d "${CONDA_PREFIX}/targets/x86_64-linux/lib" ]]; then
    die "Conda env '${ENV_NAME}' has mixed/newer CUDA libraries (CUDA 12/13). Recreate it rather than stacking packages:
  conda deactivate
  conda env remove -n ${ENV_NAME}
  bash setup.sh"
  fi
  log "Installing classic cudatoolkit=11.8 (CUDA 11 SONAMEs for CuPy; not nvidia::cuda-toolkit)"
  conda install -y "cudatoolkit=11.8"
  [[ -e "${CONDA_PREFIX}/lib/libcublas.so.11" ]] \
    || die "cudatoolkit=11.8 installed but ${CONDA_PREFIX}/lib/libcublas.so.11 is still missing"
}

install_nccl() {
  # cupy-cuda11x 12.3 looks under ~/.cupy/cuda_lib/11.x/ (NCCL 2.16.x).
  # --cuda 11.1 still works as a fallback and installs 2.8.4 for older wheels.
  log "Installing NCCL for CUDA 11.x via CuPy (idempotent)"
  if ! python -m cupyx.tools.install_library --library nccl --cuda 11.x; then
    log "NCCL 11.x install failed, falling back to --cuda 11.1"
    python -m cupyx.tools.install_library --library nccl --cuda 11.1
  fi
}

sanity_check() {
  activate_env
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.sh"
  python - <<'PY'
import os
import sys

print("Python:", sys.version.split()[0])
try:
    import torch
except Exception as exc:
    print("PyTorch import FAILED:", exc)
    sys.exit(1)

print("PyTorch:", torch.__version__)
print("CUDA compiled:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: NONE")
    sys.exit(1)

try:
    import cupy
    print("CuPy:", cupy.__version__)
    print("CuPy CUDA runtime:", cupy.cuda.runtime.runtimeGetVersion())
except Exception as exc:
    print("CuPy import FAILED:", exc)
    sys.exit(1)

try:
    import cupy.cuda.nccl as nccl
    print("NCCL available: True")
    print("NCCL version:", nccl.get_version())
except Exception as exc:
    print("NCCL import FAILED:", exc)
    print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH", ""))
    sys.exit(1)

import six  # noqa: F401
print("six: OK")
print("SANITY_CHECK_OK")
PY
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  sanity_check
  exit 0
fi

if [[ "${DATA_ONLY}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/download_data.sh"
  exit 0
fi

log "Repo root: ${ROOT}"
if is_wsl; then
  log "WSL detected. Multi-process tests should use --tensor-comm gloo."
else
  log "Native Linux assumed."
fi

require_cmd conda
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. An NVIDIA GPU + driver is required."
fi
log "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

load_conda
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda env '${ENV_NAME}' already exists"
else
  log "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} + cudatoolkit=11.8"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" "cudatoolkit=11.8"
fi
conda activate "${ENV_NAME}"
log "Using Python: $(command -v python) ($(python --version 2>&1))"
ensure_cuda11_runtime

log "Installing Python packages"
python -m pip install --upgrade pip
python -m pip install "numpy==1.24.4" "six>=1.16"

if python -c "import torch; assert torch.__version__.startswith('1.9.0')" >/dev/null 2>&1; then
  log "PyTorch 1.9.0 already installed"
else
  log "Installing PyTorch 1.9.0+cu111"
  python -m pip install "torch==1.9.0+cu111" \
    -f https://download.pytorch.org/whl/torch_stable.html
fi

if python -c "import cupy" >/dev/null 2>&1; then
  log "CuPy already importable: $(python -c 'import cupy; print(cupy.__version__)')"
else
  log "Installing CuPy 12.3.0 (cuda11x wheel; works with CUDA 11.1 runtime)"
  if ! python -m pip install "cupy-cuda11x==12.3.0"; then
    log "cupy-cuda11x failed, trying cupy-cuda110==8.6.0 (original README pin)"
    python -m pip install "cupy-cuda110==8.6.0"
  fi
fi

install_nvrtc_compat
install_nccl
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

if [[ "${SKIP_DATA}" -eq 0 ]]; then
  bash "${SCRIPT_DIR}/download_data.sh"
else
  log "Skipping dataset download"
fi

log "Running sanity check"
sanity_check

if [[ "${SKIP_SMOKE}" -eq 0 ]]; then
  log "Running 1-GPU smoke test (one tiny iteration)"
  bash "${SCRIPT_DIR}/run_1gpu_smoke.sh"
else
  log "Skipping smoke test"
fi

cat <<EOF

[setup] Done.

Next time you open a shell:
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
  source ${SCRIPT_DIR}/env.sh

Smoke tests:
  bash ${SCRIPT_DIR}/run_1gpu_smoke.sh
  bash ${SCRIPT_DIR}/run_2rank_1gpu_smoke.sh

On WSL, keep --tensor-comm gloo for 2-process runs. NCCL 2.8.4 often fails
WSL PCI topology lookup. Use NCCL on native multi-GPU Linux.
EOF
