# macOS and Apple silicon

DT-FM now accepts `--device mps` for native Apple Metal GPU compute in the
core GPipe runner, BERT QA trainer, and summarization trainer. Pipeline traffic
uses CPU-staged Gloo, so a native Mac rank can join Windows or Linux ranks.
`--device auto` in the core runner selects MPS when it is available and no
CUDA/ROCm/XPU device was selected. `--device cpu` remains available on Macs.

## Native setup

Use an Apple silicon Mac with native arm64 Python. Python 3.11 or 3.12 is a
practical starting point. Current [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
and [Apple's installation guide](https://developer.apple.com/metal/pytorch/)
require macOS 14 or newer; check the requirements for the release you install.
Install Xcode command-line tools with `xcode-select --install` if needed.

From the checkout root:

```bash
PYTHON=python3 bash scripts/setup_macos.sh
source .venv-mac/bin/activate
```

The setup creates `.venv-mac`, installs native PyTorch and the core/QA/mT5
dependencies, checks MPS and Gloo availability, then runs two synthetic
forward/backward/SGD steps. It uses PyPI's macOS wheel and requires no Conda,
CUDA, or CuPy. Use `--skip-smoke` to install and check the runtime without
training. `VENV_PATH` selects a different virtual environment.

On a CPU-only Mac, or when a supported Metal runtime is unavailable:

```bash
DEVICE=cpu bash scripts/setup_macos.sh
```

Apple silicon GPU execution requires native macOS. The Linux VM used by
Docker Desktop does not expose Apple's MPS device; Docker training uses CPU.
For a containerized ARM64 runtime, build the `:mac` image:

```bash
docker compose -f compose.yaml -f compose.mac.yaml up --build train
```

The [Docker guide](DOCKER.md#apple-silicon-docker-image) covers the standalone
image, distributed containers, MLflow, and publishing the Mac tag.
This integration targets Mac computers, including the M-series. It does not
add iPhone/iPad training or use the Apple Neural Engine.

## Verify training

With the virtual environment activated:

```bash
python scripts/run_macos_smoke.py
python scripts/run_macos_smoke.py --mixed-cpu
python scripts/run_macos_smoke.py --workflow qa --mixed-cpu
python scripts/run_macos_smoke.py --workflow summarization --mixed-cpu
```

Each command has a 120-second timeout and writes to a fresh directory under
`logs/smoke`. The mixed test uses CPU rank 0 and MPS rank 1 and exercises both
activation and gradient transfers. The script checks completed steps, finite
losses, and task checkpoints. Core metrics also record the selected backend
and forward/backward/optimizer time. Model and dataset downloads are unnecessary.
`--device cpu` runs the same checks without MPS.

For the focused tests, including native GPU training when MPS is available:

```bash
python -m unittest discover -s tests -p test_mps_support.py -v
```

## Mixed computers

Every rank must use the same checkout, compatible PyTorch releases, and
matching model/topology/batch settings. Use reachable private IPs and permit
Gloo peer traffic in both directions. Gloo opens peer sockets beyond the
coordinator port. Start these two ranks in their respective checkouts:

```bash
# Mac, rank 0. Replace this address with this Mac's LAN/Tailscale IPv4.
PYTHON=.venv-mac/bin/python DEVICE=mps RANK=0 WORLD_SIZE=2 \
  MASTER_IP=192.168.1.10 SYNTHETIC_DATA=true ITERS=2 \
  bash scripts/run_rank.sh

# Linux, rank 1. Use its own interpreter and available device.
PYTHON=python DEVICE=cpu RANK=1 WORLD_SIZE=2 \
  MASTER_IP=192.168.1.10 SYNTHETIC_DATA=true ITERS=2 \
  bash scripts/run_rank.sh
```

Windows ranks can run `dist_runner.py` directly from PowerShell using the
same settings from [SETUP.md](SETUP.md). NVIDIA ranks in a mixed pipeline must
also use `--tensor-comm gloo`; keep `--data-group-size 1` and pipeline size
equal to world size for the core runner.

The macOS launcher avoids CUDA library setup and lets Gloo select the network
interface for remote runs. Set `GLOO_SOCKET_IFNAME` to a real local interface
when necessary. `DTFM_LOCAL=1` selects macOS loopback `lo0` for local tests;
do not use loopback for peers on other computers.

## Cluster console

After setup and activation, inspect and contribute the Mac GPU:

```bash
python -m cluster_control.worker inspect
python -m cluster_control.worker pair --server http://192.168.1.20:8080 \
  --address 192.168.1.10 --name "Mac worker" --devices mps:0
python -m cluster_control.worker run
```

The worker prompts for the dashboard's one-time pairing code. Use the
control server's address for `--server` and this Mac's reachable address for
`--address`. MPS is eligible for QA/smoke jobs in the dashboard. Duplicate
agents cannot reserve the same reported Mac GPU twice. Sharing CPU and MPS
permits selecting either device; one agent per physical GPU is recommended.

## Execution limits and memory

The core GPipe MPS path uses FP32, synchronous compute, no CUDA profiling, and
pipeline-only Gloo. MPS core data parallelism is rejected. The separate QA and
summarization trainers already exchange replica gradients on CPU and accept
MPS with that transport. Their float64 loss/norm reductions run on CPU.
QA checkpoints save and restore MPS random state.
Scheduler profiling during rebalancing also preserves MPS random state.

Compute profiles and launch manifests accept `mps`. Dynamic scheduling reads
macOS available host memory using psutil and bounds MPS allocations by both
host headroom and the recommended Metal working set minus driver allocation.
CPU and MPS ranks on one Mac share the host budget. The Metal ceiling is
another constraint on that same RAM, not additional GPU memory. These values
and the model footprint are estimates; they do not guarantee a large model
will fit. Use static stage sizes when an older PyTorch build lacks MPS memory
telemetry. Start with the small synthetic test before increasing model size.

## Validation status

The implementation was checked on Windows with CPU PyTorch and mocked MPS
APIs. Real CPU/Gloo runs cover the core, QA, and mT5 paths. Native MPS tests
are skipped when a Metal GPU is unavailable. Apple GPU execution, macOS setup,
and physical Mac-to-Windows/Linux training still require validation on those
hosts. No Apple GPU throughput or model-quality result is claimed.
