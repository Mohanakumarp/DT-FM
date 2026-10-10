# BERT-Large SQuAD with Top-K gradient compression

This experiment fine-tunes `google-bert/bert-large-uncased` for extractive question
answering on SQuAD 1.1. Its QA model has 334,094,338 parameters (24 encoder
layers, width 1024). Use four NVIDIA laptops with at least 6 GB GPU memory,
Linux rank 0 at `100.79.185.101`, and Windows ranks 1, 2 and 3. These commands
reuse running Windows `dtfm-tailnet` sidecars from [the networking setup](./QA_TAILSCALE.md).

`world-size=4,pipeline-size=2` creates two two-stage replicas. Pipelines are
`0 -> 1` and `2 -> 3`; gradient synchronization pairs are `(0,2)` and `(1,3)`.
The first stage owns embeddings and encoder layers 0 through 11; the second
owns encoder layers 12 through 23 and the answer-span head.

## Compression and memory

The QA runner now accepts `--gradient-compression none|topk`. Top-K retains
AdamW and sends the largest accumulated gradient coordinates per CPU bucket,
keeping unsent coordinates in local error-feedback buffers. The default remains
dense exchange. This is Top-K with error feedback, not the momentum-SGD DGC
algorithm. Losses are normalized by the global feature count before backward;
gradient updates are summed across replicas without a second division.

The command keeps 5% of coordinates initially, exponentially reducing to 1%
over 20 optimizer steps. At 1% the FP32-values/int32-indices representation is
approximately 50 times smaller than a dense FP32 gradient representation.
These are representation sizes, not measured network traffic or a guaranteed
training speedup. Pipeline activations and backward activation gradients stay
dense. Compression delays updates and can change F1/exact match; compare with
`--gradient-compression none` using the same model, data, batch, optimizer and
training budget in a separate output directory.

For the 6 GB GPUs, begin with batch size 2 per replica, microbatch size 1 and
sequence length 256. The global batch is 4 features. The first stage has
182,937,600 parameters and the second has 151,156,738. FP32 weights, gradients
and Adam moments alone require approximately 2.73 GiB and 2.25 GiB respectively,
excluding activations, allocator/workspace overhead and other GPU applications.
AdamW uses sequential tensor operations to avoid foreach partition-sized
temporaries. Actual peak allocation is measured in each rank's summary.

Top-K additionally requires approximately 0.68 GiB and 0.56 GiB of CPU residual
memory per corresponding stage. Every host initially downloads/loads the full
model in CPU RAM. Compression does not reduce parameter, gradient or optimizer
memory. If the real run runs out of GPU memory, reduce batch size to 1 per
replica on all ranks; preserve the same batch setting in comparison runs.

## Publish the updated image once

The previous QA image has no compression flags; build and push this source
revision before running the commands. Publication is performed by the user:

```bash
sudo docker build --target cuda -t suryanarayanaant/dtfm:cuda .
sudo docker push suryanarayanaant/dtfm:cuda
```

## Linux rank 0: prepare, then train in one container

```bash
sudo docker run --pull always --rm --init --gpus all --network host \
  -e GLOO_SOCKET_IFNAME=tailscale0 \
  -e HF_HOME=/cache/huggingface \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs \
  suryanarayanaant/dtfm:cuda bash -c '
    python -u /app/scripts/train_qa_pipeline.py \
      --device cpu --prepare-only \
      --rank 0 --world-size 4 --pipeline-size 2 \
      --model google-bert/bert-large-uncased --dataset rajpurkar/squad \
      --train-examples 2000 --validation-examples 200 \
      --max-length 256 --doc-stride 64 \
    && exec python -u /app/scripts/train_qa_pipeline.py \
      --device cuda --rank 0 --world-size 4 --pipeline-size 2 \
      --dist-url tcp://100.79.185.101:29500 \
      --model google-bert/bert-large-uncased --dataset rajpurkar/squad \
      --train-examples 2000 --validation-examples 200 \
      --max-length 256 --doc-stride 64 \
      --batch-size 2 --micro-batch-size 1 --lr 0.00003 \
      --epochs 1 --max-steps 100 \
      --gradient-compression topk --compression-keep-ratio 0.01 \
      --compression-warmup-steps 20 --compression-warmup-ratio 0.05 \
      --timeout-seconds 600 --log-interval 15 \
      --output-dir logs/qa-bert-large-topk-pilot
  '
```

## Windows PowerShell ranks 1, 2 and 3

Set `$Rank` uniquely on each laptop; the remainder of the command is identical:

```powershell
$Rank = 1 # Use 2 and 3 on the other Windows laptops

docker run --pull always --rm --init --gpus all `
  --network container:dtfm-tailnet `
  -e GLOO_SOCKET_IFNAME=tailscale0 `
  -e HF_HOME=/cache/huggingface `
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 `
  -v dtfm-hf-cache:/cache -v dtfm-logs:/app/logs `
  suryanarayanaant/dtfm:cuda bash -c "
    python -u /app/scripts/train_qa_pipeline.py \
      --device cpu --prepare-only \
      --rank $Rank --world-size 4 --pipeline-size 2 \
      --model google-bert/bert-large-uncased --dataset rajpurkar/squad \
      --train-examples 2000 --validation-examples 200 \
      --max-length 256 --doc-stride 64 \
    && exec python -u /app/scripts/train_qa_pipeline.py \
      --device cuda --rank $Rank --world-size 4 --pipeline-size 2 \
      --dist-url tcp://100.79.185.101:29500 \
      --model google-bert/bert-large-uncased --dataset rajpurkar/squad \
      --train-examples 2000 --validation-examples 200 \
      --max-length 256 --doc-stride 64 \
      --batch-size 2 --micro-batch-size 1 --lr 0.00003 \
      --epochs 1 --max-steps 100 \
      --gradient-compression topk --compression-keep-ratio 0.01 \
      --compression-warmup-steps 20 --compression-warmup-ratio 0.05 \
      --timeout-seconds 600 --log-interval 15 \
      --output-dir logs/qa-bert-large-topk-pilot
  "
```

Preparation downloads weights and tokenizes data on CPU; training moves each
partition to its NVIDIA GPU. Launch all four commands promptly. Initial model
downloads can take different amounts of time; the rendezvous/communication
timeout is 600 seconds, with progress heartbeats every 15 seconds.

## Results and full training

These commands run a real-data 100-step pilot, followed by F1/exact-match
evaluation on 200 held-out questions. It establishes feasibility and initial
timings, not convergence or final model quality. To run three full epochs,
remove `--max-steps 100`, set `--epochs 3`, and choose a new output directory on
all ranks. To match the earlier larger data budget, change both preparation and
training to `--train-examples 8000 --validation-examples 1000` on every host.

Each completed step reports `gradient_keep_ratio`, `gradient_dense_bytes`,
`gradient_payload_bytes`, `gradient_payload_compression`, and
`gradient_sync_seconds`, in addition to total step time and loss. Payload bytes
describe this rank's encoded representation. They include values and indices,
but exclude protocol traffic, pipeline activations and peer replication.

Checkpoint files preserve each rank's compressor buffers and completed
compression steps. The QA runner does not yet support checkpoint resume; its
exported model contains updated model weights, not pending residuals. Progress
logs, metrics, summary and checkpoint paths follow the same volume layout as
the smaller QA experiment.

The compressed four-rank path is tested locally using a tiny BERT on CPU/Gloo,
including warm-up, replica equality, nonzero error feedback, payload reporting
and checkpoint assembly. Partition parameter counts were checked for BERT-Large
without allocating its tensors. Full BERT-Large NVIDIA/Tailscale training and
quality after compression require measurements on the physical laptops.

References: [BERT-Large](https://huggingface.co/google-bert/bert-large-uncased),
[SQuAD](https://huggingface.co/datasets/rajpurkar/squad),
[gradient compression implementation notes](./GRADIENT_COMPRESSION.md).
