# BERT-Mini question answering: six ranks over Tailscale

The `qa` command fine-tunes `prajjwal1/bert-mini` on SQuAD 1.1. It uses a
four-layer, width-256 pretrained encoder and a newly initialized answer-span
head. The default data is a seeded 8,000-question training subset and a seeded
1,000-question subset of the official validation split. Training uses overlapping
256-token windows with stride 64. Metrics count each original validation question
once, combining predictions across all its windows before exact-match/F1 scoring.

For a 334M-parameter BERT-Large run on four 6 GB NVIDIA GPUs with Top-K
gradient compression, see [the larger-model commands](./QA_BERT_LARGE_COMPRESSION.md).

This code is new. Existing Docker images do not contain `qa`; build and publish
the updated source before workers pull it. No registry publication was performed
as part of implementing this runner.

## Topology

Use `--world-size 6 --pipeline-size 3`. There are two data replicas:

| Rank | OS | Replica | Owned computation |
| --- | --- | --- | --- |
| 0 | Linux | 0 | Embeddings + encoder layer 0 |
| 1 | Windows | 0 | Encoder layers 1 and 2 |
| 2 | Windows | 0 | Encoder layer 3 + answer-span head |
| 3 | Windows | 1 | Embeddings + encoder layer 0 |
| 4 | Windows | 1 | Encoder layers 1 and 2 |
| 5 | Windows | 1 | Encoder layer 3 + answer-span head |

Pipelines are `0 -> 1 -> 2` and `3 -> 4 -> 5`. Data-gradient synchronization
pairs are `(0,3)`, `(1,4)`, `(2,5)`. Each host initially loads the full model on
CPU and then retains its stage. The partition uses computation-unit counts;
embedding-heavy endpoint memory/compute is not balanced automatically.

GPipe fills forward microbatches and drains backward microbatches in reverse
order. Matching stages synchronize dense gradients with a packed CPU/Gloo
SUM exchange, followed by pipeline-wide clipping and AdamW. Activations and
gradients are staged through CPU even when computation uses CUDA. The default
batch is 24 tokenized features per replica, so the global batch is 48 features,
not 144. Long questions/passages can create multiple windows per question.

## 1. Publish the updated images from Linux

From the updated repository root:

```bash
sudo docker build --target cuda -t suryanarayanaant/dtfm:cuda .
sudo docker push suryanarayanaant/dtfm:cuda
```

The CUDA image requires an NVIDIA GPU, a compatible host driver, and working
Docker GPU access on every training laptop. Rank 0 is the user's NVIDIA Linux
machine; ranks 1 through 5 are NVIDIA Windows laptops.

## 2. Prepare Windows networking, once per laptop

Use Docker Desktop Linux containers with the WSL 2 backend. Install/update the
NVIDIA driver and verify Docker GPU access. Each training laptop must have a
distinct Tailscale sidecar IP; the Windows host's IP is not its container IP.
If `dtfm-tailnet` already exists, start/reuse it instead of recreating it.

On each Windows laptop, set its unique rank to 1, 2, 3, 4, or 5:

```powershell
$Rank = 1

docker pull suryanarayanaant/dtfm:cuda
docker run -d --name dtfm-tailnet --restart unless-stopped `
  --cap-add NET_ADMIN --cap-add NET_RAW --device /dev/net/tun `
  --mount type=volume,source=dtfm-tailscale-state,target=/var/lib/tailscale `
  --entrypoint tailscaled tailscale/tailscale:latest `
  --state=/var/lib/tailscale/tailscaled.state `
  --socket=/var/run/tailscale/tailscaled.sock

docker exec dtfm-tailnet tailscale up --hostname "dtfm-rank$Rank" --accept-dns=false
docker exec dtfm-tailnet tailscale ip -4
docker exec dtfm-tailnet tailscale ping 100.79.185.101

docker run --rm --gpus all suryanarayanaant/dtfm:cuda `
  python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

Authenticate each sidecar in the same tailnet as rank 0 using its printed login
URL. Verify peer-to-peer connectivity between sidecars as well. Tailnet policies
and firewalls must allow TCP among all workers, including port 29500 and Gloo's
dynamic peer ports. Direct Tailscale links are preferable for this benchmark;
relayed or slow residential links can make gradient exchange dominate runtime.
Linux rank 0 must have `tailscale0` with IP `100.79.185.101`.

## 3. Cache model/data on every laptop before launching the ranks

Linux, for an NVIDIA rank 0:

```bash
sudo docker run --rm --init \
  -e HF_HOME=/cache/huggingface -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs \
  suryanarayanaant/dtfm:cuda qa --device cpu --prepare-only
```

Windows PowerShell, on each laptop:

```powershell
docker run --rm --init `
  -e HF_HOME=/cache/huggingface -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 `
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs `
  suryanarayanaant/dtfm:cuda qa --device cpu --prepare-only `
  --rank $Rank --world-size 6 --pipeline-size 3
```

Preparation uses CPU and caches weights, tokenizer, data, and tokenized windows.
Wait for `event: prepared` before beginning the distributed launch. Downloads
are excluded from the measured training window.

## 4. Linux rank 0

For an NVIDIA rank 0 with working NVIDIA Container Toolkit:

```bash
sudo docker run --rm --init --gpus all --network host \
  -e GLOO_SOCKET_IFNAME=tailscale0 -e HF_HOME=/cache/huggingface \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs \
  suryanarayanaant/dtfm:cuda qa \
  --device cuda --rank 0 --world-size 6 --pipeline-size 3 \
  --dist-url tcp://100.79.185.101:29500 \
  --model prajjwal1/bert-mini --dataset rajpurkar/squad \
  --train-examples 8000 --validation-examples 1000 \
  --max-length 256 --doc-stride 64 \
  --batch-size 24 --micro-batch-size 4 --epochs 3 \
  --timeout-seconds 300 --log-interval 15 --output-dir logs/qa-6rank
```

## 5. Windows ranks 1 through 5

Run this separately on each Windows laptop, changing only `$Rank`:

```powershell
$Rank = 1 # Set to 2, 3, 4, or 5 on the other laptops

docker run --rm --init --gpus all --network container:dtfm-tailnet `
  -e GLOO_SOCKET_IFNAME=tailscale0 -e HF_HOME=/cache/huggingface `
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 `
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs `
  suryanarayanaant/dtfm:cuda qa `
  --device cuda --rank $Rank --world-size 6 --pipeline-size 3 `
  --dist-url tcp://100.79.185.101:29500 `
  --model prajjwal1/bert-mini --dataset rajpurkar/squad `
  --train-examples 8000 --validation-examples 1000 `
  --max-length 256 --doc-stride 64 `
  --batch-size 24 --micro-batch-size 4 --epochs 3 `
  --timeout-seconds 300 --log-interval 15 --output-dir logs/qa-6rank
```

Start all six ranks within the 300-second rendezvous window. Each rank has its
own device flag; all other training/topology settings must match. A worker that
is absent or hangs causes an error instead of waiting for the former 30-minute
default. This timeout applies to communication operations, not the whole run.

Before the full experiment, add `--smoke --max-steps 2 --warmup-steps 0` to
all six commands and use `--output-dir logs/qa-smoke`. This tests the same
3-stage/2-replica topology using a tiny random BERT and synthetic spans, with
no dataset download. Smoke metrics must not be presented as SQuAD results.
For a short real-data pilot, add `--max-steps 20` on all ranks and use a separate
output directory. Remove the step cap for the three-epoch experiment.

## Logging and results

Each rank prints flushed JSON records before model loading, rank rendezvous,
dataset preparation, forward/receive/send for every microbatch, backward,
gradient synchronization, clipping, optimizer update, validation and saves.
A heartbeat prints every 15 seconds, including current phase, rank, step,
microbatch where applicable, total elapsed time and time in the current phase.
These logs do not depend on completing optimizer step 1.

Per-host volume paths under `/app/logs/qa-6rank/rankN/` contain:

- `progress.jsonl`: immediate phase records and periodic heartbeats.
- `metrics.jsonl`: completed-step losses/timing and the final summary.
- `summary.json`: completed steps, throughput, training wall time, rank-local
  peak CUDA allocated bytes and global SQuAD F1/exact match.
- `checkpoint.pt`: this stage's latest weights, optimizer state, Torch RNG state
  and optional compression buffers, saved after every optimizer step by default.
- `checkpoint-step-*.pt`: the latest two checkpoint versions for recovery.
- `config.json`, and tokenizer files on replica leaders.

Rank 0 also writes `predictions.json`. Throughput counts tokenized features,
excludes the first ten optimizer steps by default, and uses the slowest rank's
summed measured step time, excluding checkpoint I/O. Training wall time includes
all training steps and checkpoint saves, and excludes downloads, initialization
and validation. `checkpoint_seconds` reports rank-local save time. Peak GPU
allocated bytes covers this process's allocated CUDA tensors, not total device
usage. Initialization of the complete model is in CPU RAM.

To copy rank 0's results from the Linux volume:

```bash
sudo docker run --rm -v dtfm-logs:/logs suryanarayanaant/dtfm:cuda \
  tar -C /logs -czf - qa-6rank > qa-6rank-rank0.tar.gz
```

This exports only that host's volume. Collect the corresponding rank directories
from Windows hosts to assemble a model. Provide all three stages from one replica:

```bash
python scripts/assemble_mt5_checkpoint.py \
  --rank-dirs checkpoints/rank0 checkpoints/rank1 checkpoints/rank2 \
  --output-dir checkpoints/bert-mini-squad
```

Despite its historical filename, the assembler recognizes QA checkpoints and
exports `BertForQuestionAnswering` weights. Optional Top-K data-gradient
compression retains AdamW and saves CPU error-feedback buffers; see the
larger-model guide. Use a new output directory for each independent experiment.

## Checkpoints and interrupted runs

The updated runner defaults to `--checkpoint-every 1`. It saves after every
completed optimizer update, before validation, on every rank. Choose
`--checkpoint-every 5` to save every five steps, or `0` to save only before final
validation. Each rank stores its own partition in its local `dtfm-logs` volume;
all data replicas also retain their own optimizer, compression and RNG state.
Saving large-model optimizer states every step can add substantial disk I/O.

A `checkpoint_saved` record confirms that all ranks have saved that step. Writes
use temporary files and atomic replacement; the latest two versioned files are
retained. If a laptop fails while writing a new checkpoint, restart all ranks:
resume selects the newest version available on every rank, so it can fall back
to the previous saved step. The fixed-size process group cannot continue training
with one laptop missing; automatic worker recovery is not implemented.

For the user's four-rank/two-stage run, retain `--output-dir logs/qa-4rank` and
add this flag to the **training** part of every Linux/Windows command:

```text
--resume
```

Keep model, dataset, seed, batch/microbatch settings, stage count, rank count,
learning rate and compression configuration unchanged. The runner restores
weights, optimizer state, Torch RNG state, compression residuals and the next
data batch; it rejects mismatched saved settings or different training data.
`--epochs` and `--max-steps` specify total budgets, including steps already saved,
and may be increased. Each rank reloads its own checkpoint. Restoring only ranks
0 and 1 is sufficient for model export, but restarting four-rank training also
requires ranks 2 and 3's local checkpoint versions.

Resume retains progress records, trims step metrics newer than the common saved
checkpoint, and appends new step metrics. The new summary describes the restart
session and records `resumed_from_step`. Starting a fresh run in an output
directory containing checkpoints is rejected; choose a new directory instead.

This requires rebuilding/pushing `suryanarayanaant/dtfm:cuda` and pulling it on
every host. It cannot recover unsaved updates from the older runner, which only
saved checkpoints after training and validation finished. Older final-only
checkpoints can be assembled for inference, but are not resumable with this path.

## Verification

Local tests cover full-model versus partitioned logits/loss/all-parameter
gradients, answer labels in overflow windows, example-level answer scoring,
real six-process Gloo training, equal weights across replicas, global loss
normalization, progress logs, checkpoint assembly, dense/Top-K resume equivalence
and fallback when one rank lacks the latest version. A real pretrained-model
SQuAD pilot is also exercised locally. These CPU tests do not establish physical
Windows/Tailscale or NVIDIA execution, accuracy after full fine-tuning, or speedup.

```bash
python -m unittest discover -s tests -p test_qa_pipeline.py -v
```

References: [BERT-Mini model](https://huggingface.co/prajjwal1/bert-mini),
[SQuAD](https://huggingface.co/datasets/rajpurkar/squad),
[Hugging Face QA preprocessing](https://huggingface.co/docs/transformers/en/tasks/question_answering).
