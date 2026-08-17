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

# Conda CUDA 11.8 runtime (classic cudatoolkit). Must come before /opt/cuda
# (system CUDA 12/13) so CuPy resolves libcublas.so.11 / libcudart.so.11.0.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# PyTorch 1.9.0+cu111 ships libcudart-*.so.11.0 whose SONAME is still
# libcudart.so.11.0 but the contents are CUDA 11.1 (no cudaMemPoolCreate).
# Importing torch first then registers that SONAME, and CuPy 12.x (built
# against CUDA 11.2+) fails. Preload the conda 11.8 runtime so both share it.
# CUDA 11.8 runtime is backward-compatible with the cu111 wheel.
if [[ -e "${CONDA_PREFIX}/lib/libcudart.so.11.0" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libcudart.so.11.0${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# Prefer the NCCL that cupy-cuda11x 12.3 loads (CUDA 11.x / 2.16.2), then
# the older 11.1 / 2.8.4 tree, then any other CuPy-installed libnccl.so.2.
_NCCL_SO=""
for _cand in \
  "${HOME}/.cupy/cuda_lib/11.x/nccl/2.16.2/lib/libnccl.so.2" \
  "${HOME}/.cupy/cuda_lib/11.1/nccl/2.8.4/lib/libnccl.so.2" \
  "${HOME}/.cupy/cuda_lib/11.x/nccl/"*/lib/libnccl.so.2
do
  if [[ -e "${_cand}" ]]; then
    _NCCL_SO="${_cand}"
    break
  fi
done
if [[ -z "${_NCCL_SO}" ]]; then
  _NCCL_SO="$(find "${HOME}/.cupy/cuda_lib" -name 'libnccl.so.2' 2>/dev/null | head -n 1 || true)"
fi
if [[ -n "${_NCCL_SO}" ]]; then
  export LD_LIBRARY_PATH="$(dirname "${_NCCL_SO}"):${LD_LIBRARY_PATH}"
fi
unset _NCCL_SO _cand

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
