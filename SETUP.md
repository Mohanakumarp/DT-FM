# Device setup

DT-FM can run its GPipe training path on CPU, NVIDIA CUDA, AMD ROCm, Intel XPU,
or Intel UHD DirectML. CPU is the portable distributed baseline. Accelerator
data parallelism remains vendor-specific, while pipeline ranks can mix devices
through CPU-staged Gloo communication.

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

## Intel Arc and Core Ultra graphics

Use upstream PyTorch XPU on a supported Intel Arc GPU or Core Ultra processor
with Arc graphics. Older Intel UHD and Iris Xe devices are not guaranteed to
work. Install a current Intel graphics driver before running the setup:

```powershell
.\scripts\setup_intel_xpu.ps1
```

The script creates `.venv-intel`, installs PyTorch from the official XPU wheel
index, verifies device enumeration, runs a matrix multiplication with backward
propagation, and then runs one synthetic DT-FM training iteration. Repeat only
the repository test with:

```powershell
.\scripts\run_intel_xpu_smoke.ps1
```

Automatic device selection uses XPU when the installed PyTorch build reports
`torch.xpu.is_available()`. If no supported NVIDIA, AMD, or Intel accelerator is
available, `--device auto` falls through to CPU. Explicit `--device xpu` remains
strict and reports an error instead of silently falling back.

Current restrictions:

- FP32 and `--profiling no-profiling` are required.
- XPU data parallelism and XCCL are not enabled.
- Multi-process XPU is limited to pipeline-only CPU-staged Gloo.
- Intel hardware execution must be verified on a supported laptop.

## Intel UHD graphics through DirectML on WSL

Intel UHD graphics such as the UHD unit paired with the Core i5-13450HX are
not supported by the native PyTorch XPU path above. On a Windows laptop, WSL
can instead use Microsoft's `torch-directml` backend. Keep it in a separate
environment from the existing CUDA environment.

This package had an import-time failure with Python 3.8 during target-laptop
testing. Use Python 3.11:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n dtfm-directml311 python=3.11 -y
conda activate dtfm-directml311
bash scripts/setup_intel_uhd_directml.sh
```

The setup lists every DirectML adapter. On a hybrid NVIDIA plus Intel laptop,
do not assume index 0 is Intel. The tested laptop reported NVIDIA at index 0
and Intel UHD at index 1. Run the integrated smoke test with the Intel index:

```bash
DIRECTML_ID=1 EXPECTED_DIRECTML_NAME=Intel \
  bash scripts/run_intel_uhd_directml_smoke.sh
```

The runner checks the adapter name before it opens the DirectML device, so the
test stops if the selected index is not Intel. It performs adapter selection
and DT-FM training in one Python process to avoid holding two DirectML contexts
in quick succession. A direct invocation can use `--device directml
--directml-id 1 --directml-expected-name Intel`. DirectML is deliberately
excluded from `--device auto`, because choosing the correct adapter on a hybrid
laptop must be explicit.

Current restrictions:

- DirectML requires Python 3.11 in this tested configuration.
- FP32 and `--profiling no-profiling` are required.
- DirectML data parallelism is not enabled.
- Multi-process DirectML is limited to pipeline-only CPU-staged Gloo and still
  needs target-hardware verification.

The integrated smoke test passed on the target Core i5-13450HX laptop with
Intel UHD Graphics under WSL. The test used Python 3.11, PyTorch 2.4.1, FP32,
and DirectML adapter index 1. It completed two DT-FM forward, backward, and SGD
optimizer iterations, reduced the reported loss from 1.0619 to 0.9157, wrote
the rank metrics file, and exited without a DirectML allocator error.

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

- FP32 is the tested starting precision.
- RCCL and DT-FM data parallelism are not enabled on Windows.
- Multi-process ROCm is limited to pipeline-only CPU-staged Gloo.

## Heterogeneous pipeline training with Gloo

Different pipeline ranks can compute on CPU, NVIDIA CUDA, AMD ROCm, Intel XPU,
or Intel UHD DirectML. Gloo sends activations and gradients as CPU tensors, so
the ranks do not need a shared GPU collective backend. This is pipeline
parallelism only. Set `--data-group-size 1`, use FP32, and select
`--tensor-comm gloo` on every rank.

The multi-laptop launcher accepts a separate `DEVICE` value on each machine.
For a two-laptop synthetic smoke test, start rank 0 first:

```bash
DEVICE=cpu RANK=0 WORLD_SIZE=2 MASTER_IP=100.x.x.x \
  SYNTHETIC_DATA=true ITERS=2 SKIP_PROBE=true \
  bash scripts/run_rank.sh
```

On the Intel UHD laptop, activate its Python 3.11 DirectML environment and run:

```bash
DEVICE=directml DIRECTML_ID=1 EXPECTED_DIRECTML_NAME=Intel \
  RANK=1 WORLD_SIZE=2 MASTER_IP=100.x.x.x \
  SYNTHETIC_DATA=true ITERS=2 SKIP_PROBE=true \
  bash scripts/run_rank.sh
```

Use `DEVICE=rocm`, `DEVICE=xpu`, or `DEVICE=cuda` on other accelerator ranks.
Every rank must use identical model, batch, world-size, and pipeline-size
arguments. Each machine may use its own device-specific Python environment.

To exercise two devices on one Linux or WSL system, use the local launcher. Its
defaults are CPU on rank 0 and DirectML on rank 1:

```bash
DIRECTML_ID=1 EXPECTED_DIRECTML_NAME=Intel \
  bash scripts/run_mixed_local_smoke.sh
```

Set `RANK0_PYTHON`, `RANK1_PYTHON`, `RANK0_DEVICE`, and `RANK1_DEVICE` when the
ranks need different interpreters or devices. A local CPU plus Radeon 860M run
has completed two full forward, backward, and optimizer iterations successfully.

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
- mixed CPU and Radeon 860M pipeline training through CPU-staged Gloo
- deterministic synthetic smoke data
- automatic fallback to CPU when no supported accelerator is available

Verified on the target Intel Core i5-13450HX laptop under WSL:

- Intel UHD DirectML device enumeration with Python 3.11
- DT-FM model forward, backward, and two optimizer steps on Intel UHD
- integrated `dist_runner.py` DirectML smoke test with two iterations

Implemented but awaiting supported Intel hardware verification:

- single-process Intel XPU FP32 training
- mixed CPU and Intel UHD DirectML pipeline training through Gloo
- mixed CPU and Intel XPU pipeline training through Gloo

Legacy paths retained but not re-verified in this environment:

- single-process NVIDIA CUDA training
- NVIDIA pipeline communication through NCCL or Gloo

Not implemented in this tree:

- heterogeneous GPU-to-GPU collectives
- heterogeneous data parallelism
- `dist_1f1b_pipeline_async.py`
- `dist_gpipe_pipeline_async_offload.py`
