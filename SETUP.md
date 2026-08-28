# Device setup

DT-FM can run its GPipe training path on CPU, NVIDIA CUDA, or a single AMD
ROCm device. CPU is the portable distributed baseline. AMD support is currently
limited to one process while the Windows ROCm communication stack matures.

## CPU on Windows

Requirements are Python 3.10 or newer and enough RAM for the selected model.
From PowerShell in the repository root:

```powershell
.\scripts\setup_cpu.ps1
```

The script creates `.venv-cpu`, installs the official CPU PyTorch wheel, and
runs a deterministic synthetic training iteration. Run the tests again with:

```powershell
.\scripts\run_cpu_smoke.ps1
.\scripts\run_cpu_smoke.ps1 -TwoRanks
```

The two-rank test starts both processes locally and sends activations and
gradients through Gloo.

## CPU on Linux or WSL

Create a virtual environment and install the CPU build of PyTorch:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
```

Run a one-process smoke test:

```bash
python -u dist_runner.py \
  --device cpu \
  --world-size 1 --pipeline-group-size 1 --data-group-size 1 --rank 0 \
  --synthetic-data true --synthetic-vocab-size 512 \
  --seq-length 32 --embedding-dim 64 --num-layers 1 --num-heads 4 \
  --batch-size 2 --micro-batch-size 1 --num-iters 1 \
  --use-offload false --profiling no-profiling
```

For a CPU-only cluster, give every laptop the same PyTorch version and model
arguments. Use `--tensor-comm gloo`, set `--world-size` and
`--pipeline-group-size` to the number of laptops, and point every rank at the
rank-0 address with `--dist-url tcp://ADDRESS:PORT`.

## AMD Ryzen AI 7 350 and Radeon 860M

The Radeon 860M uses the `gfx1152` target. AMD publishes device-specific
PyTorch packages for it through TheRock. These packages are newer than the
production Windows ROCm stack, so keep them in a separate environment. The
tested package requires Python 3.12:

```powershell
.\scripts\setup_amd_rocm.ps1
```

The script creates `.venv-amd`, installs `torch[device-gfx1152]`, checks HIP
device enumeration, and runs one synthetic FP32 training iteration. To repeat
only the repository test:

```powershell
.\scripts\run_amd_smoke.ps1
```

ROCm PyTorch exposes AMD GPUs through `torch.cuda`. DT-FM distinguishes ROCm
by checking `torch.version.hip`. CuPy NCCL is not loaded for this path.

Current restrictions:

- AMD runs require `--world-size 1`.
- FP32 is the tested starting precision.
- RCCL and DT-FM data parallelism are not enabled on Windows.
- Use the CPU environment for multi-laptop training.

## NVIDIA CUDA (legacy path)

The original CUDA 11 setup remains in the repository, but it was not re-tested
as part of the CPU and Radeon work:

```bash
bash setup.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtfm
source scripts/env.sh
bash scripts/run_1gpu_smoke.sh
```

The legacy setup installs PyTorch 1.9, CuPy, CUDA 11.8, and NCCL. Use
`--tensor-comm gloo` for two processes sharing one GPU or when NCCL topology
detection fails under WSL.

## QQP data

Synthetic smoke tests do not require external data. Download QQP before a real
training run:

```bash
bash scripts/download_data.sh
```

Expected files:

```text
task_datasets/data/bert-large-cased-vocab.txt
task_datasets/data/QQP/train.tsv
task_datasets/data/QQP/dev.tsv
task_datasets/data/QQP/test.tsv
```

The QQP TSV files are ignored by Git.

## Supported scope

Verified in the current Windows test environment:

- single-process CPU training
- CPU pipeline parallelism over Gloo
- single-process AMD ROCm training
- deterministic synthetic smoke data

Legacy paths retained but not re-verified in this environment:

- single-process NVIDIA CUDA training
- NVIDIA pipeline communication through NCCL or Gloo

Not implemented in this tree:

- Intel XPU execution
- AMD multi-process GPU execution
- heterogeneous GPU-to-GPU collectives
- `dist_1f1b_pipeline_async.py`
- `dist_gpipe_pipeline_async_offload.py`
