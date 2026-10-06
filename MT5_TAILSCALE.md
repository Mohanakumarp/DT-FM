# Image-only training over Tailscale: Linux and Windows

First publish the updated configurable runner as described in
[DISTRIBUTED_SUMMARIZATION.md](./DISTRIBUTED_SUMMARIZATION.md). The earlier image
does not support these new options. Workers need no ZIP or source checkout.
Model, dataset, and rank-count changes are runtime options after that update.

Linux rank 0 is `100.79.185.101`. Set `N` to any positive total rank count for
the run, with unique ranks `0` through `N-1`. All ranks must agree on training
settings. Joining/leaving during a run requires a restart.

## One-time setup on each Windows laptop

These examples use NVIDIA GPUs and the `:cuda` image. On Windows, install a
current NVIDIA driver, update WSL with `wsl --update`, and use Docker Desktop's
WSL 2 backend and Linux containers. See [Docker GPU requirements](https://docs.docker.com/desktop/features/gpu/). The training container
shares a Tailscale container's network, giving Gloo a real `tailscale0` interface
and reachable peer addresses without Docker Desktop host networking.

In PowerShell, choose this laptop's rank:

```powershell
$Rank = 1 # Use 2, 3, ... on other laptops

docker pull suryanarayanaant/dtfm:cuda
docker run -d --name dtfm-tailnet --restart unless-stopped `
  --cap-add NET_ADMIN --cap-add NET_RAW --device /dev/net/tun `
  --mount type=volume,source=dtfm-tailscale-state,target=/var/lib/tailscale `
  --entrypoint tailscaled tailscale/tailscale:latest `
  --state=/var/lib/tailscale/tailscaled.state `
  --socket=/var/run/tailscale/tailscaled.sock

docker exec dtfm-tailnet tailscale up --hostname "dtfm-rank$Rank" --accept-dns=false
```

Open the printed login URL and authenticate into the same tailnet as Linux rank 0.
The container is an additional Tailscale device with its own IP, distinct from the
Windows host. State persists in the named volume. In later sessions, use
`docker start dtfm-tailnet`, not the create command. Rank assignment can change
between runs without recreating the sidecar.

Check connectivity on each Windows laptop:

```powershell
docker exec dtfm-tailnet tailscale ip -4
docker exec dtfm-tailnet tailscale ping 100.79.185.101
```

Tailnet policies/firewalls must permit TCP among all training nodes, including
29500 and Gloo's dynamic peer ports. Test between worker container IPs too.
Windows-host Tailscale or forwarding only 29500 does not establish this network.
Linux rank 0 must have `tailscale0` and `tailscale ip -4` must show the coordinator
IP. This setup has not been run here on physical Windows laptops; smoke-test first.

## Check GPU access before training

Linux requires a compatible NVIDIA driver and the configured
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Run this on every NVIDIA laptop after pulling the updated `:cuda` image
(prefix with `sudo` on Linux if needed):

```text
docker run --rm --gpus all suryanarayanaant/dtfm:cuda python -c "import torch; x = torch.ones(1, device='cuda'); print(torch.cuda.get_device_name(0), x.device)"
```

It must print the GPU name and `cuda:0`. The training commands explicitly request
CUDA, so unavailable GPU access causes an error. The GPU image defaults to CUDA.
Computation runs on the GPU; transfers between laptops still use CPU-staged Gloo.

For a CPU-only laptop in a mixed run, use `:cpu`, remove `--gpus all`, and change
`--device cuda` to `--device cpu`. Keep model/data/topology flags consistent.
AMD/Intel GPUs are not supported by the NVIDIA `:cuda` image.

## Linux rank 0

```bash
N=3 # Choose the total rank count for this run
sudo docker pull suryanarayanaant/dtfm:cuda
sudo docker run --rm --init --gpus all --network host \
  -e GLOO_SOCKET_IFNAME=tailscale0 -e HF_HOME=/cache/huggingface \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs \
  suryanarayanaant/dtfm:cuda summarize \
  --device cuda --world-size "$N" --rank 0 --dist-url tcp://100.79.185.101:29500 \
  --model google/mt5-small --dataset csebuetnlp/xlsum --dataset-config tamil \
  --source-column text --target-column summary \
  --batch-size 4 --micro-batch-size 1 --steps 1000
```

Other Linux laptops use their own rank instead of 0.

## Every Windows worker

```powershell
$N = 3    # Same total as Linux rank 0
$Rank = 1 # Unique rank from 1 through N-1

docker run --rm --init --gpus all --network container:dtfm-tailnet `
  -e GLOO_SOCKET_IFNAME=tailscale0 -e HF_HOME=/cache/huggingface `
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 `
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs `
  suryanarayanaant/dtfm:cuda summarize `
  --device cuda --world-size $N --rank $Rank --dist-url tcp://100.79.185.101:29500 `
  --model google/mt5-small --dataset csebuetnlp/xlsum --dataset-config tamil `
  --source-column text --target-column summary `
  --batch-size 4 --micro-batch-size 1 --steps 1000
```

Start every rank while rank 0 is waiting. For a network check, add `--smoke`,
change `--steps 1000` to `--steps 2`, and add `--validation-batches 1` on all ranks.
Downloads persist in `dtfm-hf-cache`; checkpoints/metrics persist in `dtfm-logs`.

Change model/dataset/language/column flags on every rank for another experiment.
No source mount, rebuild, or push is required for these runtime settings.
See the main guide for model support, pipeline sizes, and checkpoint export.

Sources: [Tailscale Docker kernel networking](https://tailscale.com/docs/features/containers/docker/docker-params),
[Tailscale CLI](https://tailscale.com/docs/reference/tailscale-cli),
[Docker shared network namespaces](https://docs.docker.com/engine/network/#container-networks).
