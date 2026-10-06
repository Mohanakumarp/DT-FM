# Running DT-FM in Docker

Docker packages the Python environment and the repository's training, scheduler,
profiling, reporting, dataset, and MLflow tools. The default image uses CPU;
the `cuda` target adds NVIDIA CUDA and CuPy/NCCL. Downloaded data is kept outside
the image, so builds do not need datasets or a GPU.

## Quick start

Install Docker Engine with Compose v2 on Linux, or Docker Desktop using Linux
containers on Windows/macOS.

Prebuilt images are published to [Docker Hub](https://hub.docker.com/r/suryanarayanaant/dtfm):
`suryanarayanaant/dtfm:cpu` (also `:latest`) and `suryanarayanaant/dtfm:cuda`.
These published images target `linux/amd64`. ARM machines can build the CPU
image locally with the command below.

Run the CPU smoke test without cloning or building the project:

```bash
docker run --rm -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 suryanarayanaant/dtfm:cpu
```

From the repository root, pull the published image and run Compose with its
persistent volumes:

```bash
export DTFM_IMAGE=suryanarayanaant/dtfm
docker compose pull train
docker compose up --no-build train
```

The same `DTFM_IMAGE` setting selects the published images for distributed
workers, MLflow, and the GPU service. For example, `docker compose pull rank0
rank1` followed by `docker compose up --no-build rank0 rank1` runs two workers.
Use `docker compose pull gpu` followed by `docker compose up --no-build gpu`
on a host with NVIDIA Container Toolkit and a compatible GPU driver.

Alternatively, build from source (including native ARM CPU images):

```bash
unset DTFM_IMAGE
docker compose up --build train
```

On Linux, `permission denied` for `/var/run/docker.sock` means your shell needs
Docker access. Run the commands with `sudo` or use a Docker-enabled account.

This performs two actual forward/backward/optimizer steps using a small synthetic
model, then exits. It needs no QQP download. Metrics are saved to
`logs/smoke/metrics_rank0.json`. The CPU image supports `linux/amd64` and
`linux/arm64`, including Apple Silicon via Docker Desktop; builds select the
host architecture. Allocate enough Docker memory for your chosen model.
Multiline shell examples below use Bash; on Windows, use WSL/Git Bash or put
the command on one line.

The images use Python 3.10, PyTorch 2.8.0, and the direct dependency pins in
`docker/requirements.txt`. Transitive Python dependencies and base image tags
are resolved at build time. Builds need access to the image registry, PyPI,
PyTorch's wheel index, and the OS package repositories. After building, the
synthetic smoke tests work offline.

## Distributed training and checks

Run two pipeline ranks in separate containers on the Compose network:

```bash
docker compose up --build rank0 rank1
```

Both ranks exit after two optimizer steps; results are in
`logs/compose-2rank/metrics_rank0.json` and `metrics_rank1.json`. Gloo uses `eth0`
and Compose DNS resolves `rank0`. No training ports need publishing for this
single-machine network. `--dist-timeout-seconds 60` bounds process-group waits.

Alternatively, spawn two ranks inside one container, run the existing tests,
or verify the communication probe:

```bash
docker compose run --rm -e DTFM_WORLD_SIZE=2 train
docker compose run --rm train test
docker compose run --rm train python scripts/verify_comm_probe.py --world-size 2 --output-dir /app/logs/probe
```

The Docker CI workflow builds both images, runs the existing tests, checks
training between two containers, and runs the MLflow demo. CUDA CI checks
library imports; executing GPU training requires a GPU runner.

`DTFM_THREADS` sets the CPU threads per process in Compose (default 2). The
repository's Python tools are available, for example:

```bash
docker compose run --rm train python scripts/profile_compute.py --help
docker compose run --rm train python scripts/lab_demo.py --help
docker compose run --rm train bash
```

Commands that expect a particular working directory can use `--workdir`, e.g.
`docker compose run --rm --workdir /app/scheduler/heuristic_evolutionary_solver train python scheduler.py`.

Use `train` followed by `dist_runner.py` arguments for a completely custom run.
Use `train` or `smoke` to launch training in the container. The native Conda
and device installation wrappers remain host setup commands.
`smoke` accepts additional runner arguments, which override its small-model
defaults. For example, this runs ten steps on four locally spawned ranks:

```bash
docker compose run --rm -e DTFM_WORLD_SIZE=4 train smoke --num-iters 10
```

## Datasets and custom training

For runtime-selected summarisation models, datasets, and any positive total rank
count, use [the configurable summarisation workflow](./DISTRIBUTED_SUMMARIZATION.md).
The `summarize` entry point (`mt5` alias) supports CPU or NVIDIA computation,
mT5 block partitions with data replicas, and other model families through
full-model data parallelism. Workers can use the image alone.

Download QQP and its tokenizer vocabulary into the persistent `datasets` volume:

```bash
docker compose run --rm train bash scripts/download_data.sh
docker compose run --rm train smoke --synthetic-data false --num-iters 10
```

The second command uses the smoke model dimensions with the real QQP vocabulary.
For your own model, specify all desired dimensions and training options:

```bash
docker compose run --rm train train \
  --device cpu --tensor-comm gloo --dist-backend gloo \
  --world-size 1 --pipeline-group-size 1 --data-group-size 1 --rank 0 \
  --seq-length 128 --embedding-dim 128 --num-layers 2 --num-heads 4 \
  --batch-size 4 --micro-batch-size 1 --num-iters 20 \
  --use-offload false --profiling no-profiling --metrics-dir ./logs/qqp
```

To use host data instead, add a bind mount (absolute host paths work from any
directory):

```bash
docker compose run --rm -v /absolute/path/to/data:/app/task_datasets/data train smoke --synthetic-data false
```

Logs and traces are bind mounted into `./logs` and `./trace_json`; Hugging Face
downloads use the persistent `cache` volume. Images run as root to support
fresh named volumes and arbitrary bind mounts. On Linux, run with
`--user "$(id -u):$(id -g)" -e HOME=/tmp` when using host directories you own;
ensure any mounted named volumes are also writable by that UID. Host-mounted
output from the default root user may require elevated permissions to edit.

## MLflow

Start the tracking server and wait for its health check:

```bash
docker compose up --build -d --wait mlflow
docker compose run --rm tracking-demo
```

Open <http://localhost:5000> and select **DT-FM demo**. The demo runs two CPU
trials and a two-rank pipeline trial. Its dependency waits for MLflow to become
healthy before starting training. Server metadata and artifacts live in the
persistent `mlflow` volume. Set `MLFLOW_PORT` to change the host port.

Enable tracking for another training command by adding
`--mlflow-tracking-uri http://mlflow:5000`, for example:

```bash
docker compose run --rm train smoke --mlflow-tracking-uri http://mlflow:5000
```

The published UI listens on host loopback. For remote access, use an SSH tunnel
or configure the published address and authentication for your environment.

## NVIDIA GPUs

On a supported NVIDIA host, install a driver compatible with CUDA 12.6 and
configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Then build and run the GPU service:

```bash
docker compose build gpu
docker compose run --rm gpu
docker compose run --rm gpu python -c "import torch, cupy; print(torch.cuda.get_device_name(0)); print(cupy.cuda.nccl.get_version())"
```

The `cuda` image uses NVIDIA's CUDA 12.6.3 base, the official PyTorch 2.8.0
`cu126` wheel, and CuPy 13.6.0. It includes NVRTC/runtime headers and registers
the PyTorch-provided CUDA libraries, including NCCL, with the dynamic linker
for CuPy. It does not source the native
Conda `scripts/env.sh`, which configures the older CUDA 11 environment.
The GPU smoke command requests CUDA explicitly, so missing GPU access causes
an error. Two locally spawned GPU ranks can use Gloo, or homogeneous NVIDIA
ranks can use CuPy/NCCL:

```bash
docker compose run --rm -e DTFM_WORLD_SIZE=2 gpu
docker compose run --rm -e DTFM_WORLD_SIZE=2 gpu smoke --tensor-comm nccl --dist-backend cupy_nccl
```

All visible GPUs are exposed; use runner `--cuda-id` arguments or restrict
visibility to choose a device. CPU is the portable fallback for AMD/Intel hosts.
The repository's native ROCm, XPU, and WSL DirectML setups are described in
[SETUP.md](SETUP.md); those vendor runtimes are not included in these images.
Docker cannot provide missing hardware, host drivers, or DirectML device access.

## Multiple machines

Build/load the same image on every host and use matching model and PyTorch
versions. Compose's `rank0` DNS name works only within its local network. On
Linux, use host networking and the real reachable rank-0 IP. Gloo opens peer
ports in addition to the rendezvous port; publishing only port 9000 is
insufficient. Allow peer traffic on your LAN or VPN and set the interface
actually connecting the hosts.

For example, run on rank 0 and rank 1 respectively, replacing `192.168.1.10`
and `eth0` with the rank-0 address and each host's own interface:

```bash
docker build --target cpu -t dtfm:cpu .
docker run --rm --init --network host --shm-size 1g \
  -e OMP_NUM_THREADS=2 -e GLOO_SOCKET_IFNAME=eth0 \
  -v "$PWD/logs:/app/logs" dtfm:cpu smoke \
  --spawn-local-ranks false --world-size 2 --pipeline-group-size 2 \
  --dist-url tcp://192.168.1.10:9000 --rank 0
```

```bash
docker run --rm --init --network host --shm-size 1g \
  -e OMP_NUM_THREADS=2 -e GLOO_SOCKET_IFNAME=eth0 \
  -v "$PWD/logs:/app/logs" dtfm:cpu smoke \
  --spawn-local-ranks false --world-size 2 --pipeline-group-size 2 \
  --dist-url tcp://192.168.1.10:9000 --rank 1
```

Host networking on Docker Desktop has different requirements; use Linux hosts
for the above cluster recipe. Multiple containers on one physical machine must
share the same `--scheduler-host-id` when using dynamic scheduling, so available
memory is not counted twice. The lab control server, communication probe, and
log hub also need reachable ports when used across machines; their options and
ports are documented in the existing guides. SSH orchestration scripts still
need host addresses and credentials mounted at runtime. Traffic shaping (`tc`)
and strongSwan/IPsec VPN setup operate on host networking and require separate
host configuration; these commands are not run automatically by the containers.

## Moving images and cleanup

No registry account is needed to transfer an already-built image:

```bash
docker save dtfm:cpu -o dtfm-cpu.tar
# On another machine with the same CPU architecture:
docker load -i dtfm-cpu.tar
docker run --rm --init -e OMP_NUM_THREADS=2 dtfm:cpu
```

To publish both CPU architectures to a registry you control:

```bash
docker buildx build --target cpu --platform linux/amd64,linux/arm64 \
  -t YOUR_REGISTRY/dtfm:cpu --push .
```

Stop services with `docker compose down`; named volumes and host logs survive.
`docker compose down -v` also deletes downloaded datasets, caches, and MLflow
metadata/artifacts. Logs in host directories survive either command.
