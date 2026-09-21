#!/usr/bin/env bash
# Linux bootstrap. Installs a private Python environment; does not require sudo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${ROOT}/.lab-tools"
mkdir -p "${TOOLS}"
if [[ ! -x "${TOOLS}/uv" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/0.12.15/install.sh -o "${TOOLS}/install-uv.sh"
  elif command -v wget >/dev/null 2>&1; then
    wget -q https://astral.sh/uv/0.12.15/install.sh -O "${TOOLS}/install-uv.sh"
  else
    echo 'Setup needs curl or wget to download Python. Ask the lab administrator to install one.' >&2
    exit 1
  fi
  UV_UNMANAGED_INSTALL="${TOOLS}" sh "${TOOLS}/install-uv.sh"
fi
export UV_PYTHON_INSTALL_DIR="${TOOLS}/python"
export PYTHONUTF8=1
"${TOOLS}/uv" python install 3.12.10
if [[ ! -x "${ROOT}/.venv-lab/bin/python" ]]; then
  "${TOOLS}/uv" venv --python 3.12.10 --seed "${ROOT}/.venv-lab"
fi
PY="${ROOT}/.venv-lab/bin/python"
"${PY}" -m pip install --disable-pip-version-check torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
"${PY}" -m pip install --disable-pip-version-check numpy==2.2.6 six==1.17.0 psutil==7.0.0 matplotlib==3.10.3
"${PY}" "${ROOT}/scripts/lab_data.py" --rows "${LAB_ROWS:-4096}"
cd "${ROOT}"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PY}" -u dist_runner.py \
  --device cpu --spawn-local-ranks true --tensor-comm gloo --dist-backend gloo \
  --dist-url tcp://127.0.0.1:9021 --world-size 2 --pipeline-group-size 2 --data-group-size 1 \
  --rank 0 --synthetic-data true --synthetic-samples 8 --synthetic-vocab-size 512 \
  --seq-length 32 --embedding-dim 64 --num-layers 1 --num-heads 4 \
  --batch-size 2 --micro-batch-size 1 --num-iters 2 --metrics-dir ./logs/lab/setup-smoke \
  --skip-comm-probe true --use-offload false --profiling no-profiling
echo 'Setup complete. See LAB_DEMO.md; start rank 0 first.'
