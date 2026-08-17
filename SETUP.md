# New-system setup

This repository trains a small GPT-style model on GLUE QQP with GPipe pipeline parallelism. The original project targeted a multi-node AWS GPU cluster. This document is the **local reproduction path** that works on a single NVIDIA GPU, including WSL.

## What you need

- Linux or WSL2
- An NVIDIA GPU + driver (`nvidia-smi` works)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- About 5 GB disk for the conda env, NCCL, and QQP

Tested combination:

| Component | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 1.9.0+cu111 |
| CUDA runtime | 11.8 (classic `cudatoolkit=11.8` from defaults) |
| CuPy | 12.3.0 (`cupy-cuda11x`) |
| NCCL | installed by `python -m cupyx.tools.install_library --library nccl --cuda 11.1` |

The driver can be much newer than 11.8 (for example 595.x / CUDA 13.2). Do not upgrade PyTorch or CuPy to match the driver.

**Do not** run `conda install -c nvidia cuda-toolkit=11.8`. That hyphenated metapackage does not pin `cuda-version`, and the solver will pull CUDA 12/13 libraries (`libcublas.so.13`, `cuda-nvcc` 13.x, …). CuPy then fails with `libcudart.so.11.0` / `libcublas.so.11` missing. The correct package name is the classic **`cudatoolkit=11.8`** (no hyphen) from `defaults`.

## One-command install

From the repo root:

```bash
bash setup.sh
```

That script will:

1. Create conda env `dtfm` with Python 3.8
2. Install PyTorch 1.9.0+cu111, CuPy, `six`, `numpy`
3. Install NCCL 2.8.4 for CUDA 11.1
4. Add a `libnvrtc.so.11.0` compatibility symlink if needed
5. Download GLUE QQP and the BERT-large-cased vocab
6. Run a 1-GPU smoke test (`world-size=1`, 1 tiny iteration)

Useful flags:

```bash
bash setup.sh --skip-smoke
bash setup.sh --data-only
bash setup.sh --check
```

## Every new shell

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtfm
source scripts/env.sh
```

`scripts/env.sh` sets `LD_LIBRARY_PATH` so CuPy can find NCCL and the CUDA 11.8 runtime. It also `LD_PRELOAD`s conda's `libcudart.so.11.0`. That is required because PyTorch 1.9.0+cu111 ships a CUDA 11.1 `libcudart` with the same SONAME; importing torch first would otherwise shadow the 11.8 library and CuPy 12.3 fails with `undefined symbol: cudaMemPoolCreate`.

## Smoke tests

One process, one GPU (full forward / QQP loss / backward / optimizer):

```bash
bash scripts/run_1gpu_smoke.sh
```

Two processes on the **same** GPU. Rank 0 is the first pipeline stage and spawns rank 1 as the last stage:

```bash
bash scripts/run_2rank_1gpu_smoke.sh
```

On a **single GPU** (this laptop, or WSL) keep `--tensor-comm gloo`. NCCL 2.16 refuses two ranks on the same CUDA device (`Duplicate GPU detected`) and then aborts. On native **multi-GPU** Linux, run:

```bash
TENSOR_COMM=nccl bash scripts/run_2rank_1gpu_smoke.sh
```

Do **not** start rank 1 yourself when using `--spawn-local-ranks true`.

## Data (not in git)

QQP is large (~50 MB per split). It is gitignored and downloaded by `scripts/download_data.sh` from:

- QQP: `https://dl.fbaipublicfiles.com/glue/data/QQP.zip`
- Vocab: Hugging Face `bert-large-cased` `vocab.txt`

Expected layout after download:

```text
task_datasets/data/bert-large-cased-vocab.txt
task_datasets/data/QQP/train.tsv
task_datasets/data/QQP/dev.tsv
task_datasets/data/QQP/test.tsv
```

## Pushing to GitHub

`.gitignore` already excludes:

- `__pycache__/`
- `*.orig`, `*.backup`, `*.save`
- `task_datasets/data/QQP/`
- profiling JSON under `trace_json/`

Do not commit the QQP TSVs. After clone, run `bash setup.sh` (or `bash scripts/download_data.sh`).

## What is implemented vs original paper code

Working locally:

- GPipe (`--pp-mode gpipe`)
- QQP sequence classification
- `world-size=1` full model on one GPU
- `world-size=2` local pipeline via spawned rank 1
- Gloo process group + optional Gloo tensor transport

Not in this git tree (do not invent them):

- `dist_1f1b_pipeline_async.py`
- `dist_gpipe_pipeline_async_offload.py`

Known WSL limit: CuPy NCCL communicator init can fail even after `libnccl.so.2` is installed. Use Gloo tensor comm for local multi-process tests until you have native Linux + working NCCL.

## Manual install (if you do not want the script)

```bash
conda create -y -n dtfm python=3.8 cudatoolkit=11.8
conda activate dtfm
python -m pip install --upgrade pip
python -m pip install numpy==1.24.4 six
python -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install cupy-cuda11x==12.3.0
python -m cupyx.tools.install_library --library nccl --cuda 11.1
ln -sfn "$CONDA_PREFIX/lib/libnvrtc.so.11.1" "$CONDA_PREFIX/lib/libnvrtc.so.11.0"
bash scripts/download_data.sh
source scripts/env.sh
```

The PyTorch `+cu111` wheel does **not** ship `libcublas.so.11` (it only has a hashed `libcudart-*.so.11.0`). CuPy needs the unmangled CUDA 11 SONAMEs from `cudatoolkit=11.8`.
