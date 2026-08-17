#!/usr/bin/env bash
# Runtime environment for DT-FM.
# Usage (after: conda activate dtfm):
#   source scripts/env.sh

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _DTFM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _DTFM_SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
export DTFM_ROOT="$(cd "${_DTFM_SCRIPTS_DIR}/.." && pwd)"
unset _DTFM_SCRIPTS_DIR

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "env.sh: CONDA_PREFIX is empty. Activate the conda env first:"
  echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
  echo "  conda activate dtfm"
  return 1 2>/dev/null || exit 1
fi

# Conda CUDA runtime libs (libnvrtc, etc.)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CuPy-installed NCCL (usually ~/.cupy/cuda_lib/<cuda>/nccl/<ver>/lib)
_NCCL_SO="$(find "${HOME}/.cupy/cuda_lib" -name 'libnccl.so.2' 2>/dev/null | head -n 1 || true)"
if [[ -n "${_NCCL_SO}" ]]; then
  export LD_LIBRARY_PATH="$(dirname "${_NCCL_SO}"):${LD_LIBRARY_PATH}"
fi
unset _NCCL_SO

# CuPy 12 + CUDA 11.1 may ask for libnvrtc.so.11.0 while conda ships 11.1.
if [[ -e "${CONDA_PREFIX}/lib/libnvrtc.so.11.1" && ! -e "${CONDA_PREFIX}/lib/libnvrtc.so.11.0" ]]; then
  ln -sfn libnvrtc.so.11.1 "${CONDA_PREFIX}/lib/libnvrtc.so.11.0"
fi

# Localhost 2-process tests work over loopback. Override if needed.
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  export DTFM_WSL=1
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
else
  export DTFM_WSL=0
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DTFM_ENV_LOADED=1
