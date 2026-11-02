# Pair and run cluster workers in Docker

The dashboard's **Connect computer** dialog generates standalone Docker commands
for Linux or Docker Desktop and CPU or NVIDIA compute. Enter the worker's
Tailscale address and computer name, complete Docker setup, then generate a
pairing code. The dialog includes copy controls and persistent Docker volumes;
contributors using it do not need the Compose files below.

Use Docker for **every worker in the job**, including Linux rank 0. The dashboard
can keep running as it is. Containers run the code included in the image and
need no host Python environment or source mount. Pairing state, job logs, and
model downloads persist in Docker volumes.

The worker image defaults to `suryanarayanaant/dtfm:latest`. Publish the updated
image with `cluster_control` before using these commands. This guide does not
establish which source revision is currently published. The Dockerfile currently
uses Python 3.10 and PyTorch 2.8.0 for both CPU and CUDA targets.

Contributors only need Docker and these files, copied into the same folder:

- `compose.cluster.yaml`
- `compose.cluster.desktop.yaml` for Windows/macOS Docker Desktop
- `compose.cluster.gpu.yaml` for NVIDIA compute, optionally

Run commands from that folder. Keep its name/project and the Compose file list
consistent so pairing and running use the same named volumes. Do not use
`docker compose down -v`, which would delete the saved identity and caches.

## Linux worker at 100.79.185.101

Stop the native Python worker first and cancel its old job. Leave the dashboard
running. Tailscale on this Linux host must already have `100.79.185.101` on
`tailscale0`. Other hosts must advertise their own reachable addresses.

Pull the image and inspect its runtime:

```bash
docker compose -f compose.cluster.yaml pull worker
docker compose -f compose.cluster.yaml run --rm --no-deps --entrypoint python worker -c "import sys, torch; print(sys.version); print(torch.__version__)"
docker compose -f compose.cluster.yaml run --rm --no-deps worker inspect
```

Generate a fresh pairing code in the dashboard, then pair. Paste the code at
the private prompt:

```bash
docker compose -f compose.cluster.yaml run --rm --no-deps worker pair --server http://100.79.185.101:8080 --address 100.79.185.101 --name karthick_js --devices all --state /state/worker.json
```

Start contributing, or use `up -d worker` to run in the background:

```bash
docker compose -f compose.cluster.yaml up worker
```

The default command runs `python -m cluster_control.worker run` with the saved
identity, persistent output, and `--interface tailscale0`.

## Windows workers

Use Docker Desktop with Linux containers and the WSL 2 backend. A Tailscale
sidecar gives the worker a routable Linux interface, including Gloo's dynamic
peer ports. Publishing only port 29500 is insufficient. Each sidecar has its
own Tailscale IP, different from the Windows host's IP.
The sidecar uses [Tailscale kernel networking](https://tailscale.com/docs/concepts/userspace-networking)
through `/dev/net/tun` and shares that network namespace with the worker.

In PowerShell, pull the images and start the sidecar:

```powershell
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml pull
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml up -d tailscale
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml exec tailscale tailscale up --hostname dtfm-karthick-worker --accept-dns=false
```

Use a unique hostname on each computer. Open the printed login URL and join
the same tailnet as the dashboard and other workers. Then get this worker's IP
and test connectivity:

```powershell
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml exec tailscale tailscale ip -4
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml exec tailscale tailscale ping 100.79.185.101
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml run --rm --no-deps --entrypoint python worker -c "import sys, torch; print(sys.version); print(torch.__version__)"
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml run --rm --no-deps worker inspect
```

Set `$WorkerIP` to the IPv4 printed by the sidecar, then use a fresh dashboard
pairing code. Change the name for each contributor:

```powershell
$WorkerIP = "<sidecar-Tailscale-IPv4>"
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml run --rm --no-deps worker pair --server http://100.79.185.101:8080 --address $WorkerIP --name karthick_js --devices all --state /state/worker.json
docker compose -f compose.cluster.yaml -f compose.cluster.desktop.yaml up worker
```

The sidecar must remain running while the worker runs. Tailnet policy and host
firewalls must permit TCP traffic between all worker IPs, including 29500 and
Gloo's dynamic peer ports. On macOS Docker Desktop, use the same sidecar setup;
the image uses CPU compute because Docker cannot expose Apple MPS.

## NVIDIA GPUs

The repository's Docker instructions currently describe `latest` as the CPU image.
For NVIDIA compute, append `-f compose.cluster.gpu.yaml` to **every** Compose
command, including pull, inspect, pair, and up. This selects `:cuda` and requests
GPU access. Install a compatible NVIDIA host driver and, on Linux, NVIDIA
Container Toolkit. Check `worker inspect` lists your CUDA device before pairing.

For example, Linux:

```bash
docker compose -f compose.cluster.yaml -f compose.cluster.gpu.yaml pull worker
docker compose -f compose.cluster.yaml -f compose.cluster.gpu.yaml run --rm --no-deps worker inspect
docker compose -f compose.cluster.yaml -f compose.cluster.gpu.yaml run --rm --no-deps worker pair --server http://100.79.185.101:8080 --address 100.79.185.101 --name karthick_js --devices all --state /state/worker.json
docker compose -f compose.cluster.yaml -f compose.cluster.gpu.yaml up worker
```

For Windows, use all three files, in this order: `compose.cluster.yaml`,
`compose.cluster.desktop.yaml`, `compose.cluster.gpu.yaml`.

## Keep the images aligned

Pull on every machine after publication. Compare the runtime check above and
use images built from the same application revision. CPU/CUDA builds report
different build suffixes, but must have the same PyTorch release, currently
2.8.0. A moving `latest` tag can differ across machines that pulled at different
times. For a reproducible job, set `DTFM_WORKER_IMAGE` to an immutable registry
digest before pulling, pairing, and running:

```bash
export DTFM_WORKER_IMAGE=suryanarayanaant/dtfm@sha256:<published-digest>
```

PowerShell:

```powershell
$env:DTFM_WORKER_IMAGE = "suryanarayanaant/dtfm@sha256:<published-digest>"
```

Use the CPU digest for CPU workers and the matching release's CUDA digest for
GPU workers. New images take effect when the worker container is recreated;
stop active jobs before updating. Pair once per saved identity. Later restarts
only need `up worker`; a second pair command deliberately refuses to overwrite
`/state/worker.json`. Stop the old native worker and remove its stale dashboard
device entry to avoid accidentally scheduling it alongside the Docker worker.

The volume `worker-state` contains the pairing token and artifacts. Each host
must keep its own identity. Never copy that volume between contributors.
