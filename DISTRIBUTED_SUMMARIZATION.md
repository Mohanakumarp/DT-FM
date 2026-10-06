# Configurable distributed summarisation

The `summarize` Docker command selects models, datasets, and laptop counts at
runtime. `mt5` remains an alias. Workers need only the image, with no repository
checkout or source ZIP.

## Publish the updated runner once

Your earlier image contains the fixed-rank runner. Publish this update once:

```bash
sudo docker build --target cuda -t suryanarayanaant/dtfm:cuda .
sudo docker push suryanarayanaant/dtfm:cuda
# Optional CPU runtime for laptops without NVIDIA GPUs:
sudo docker build --target cpu -t suryanarayanaant/dtfm:cpu .
sudo docker push suryanarayanaant/dtfm:cpu
```

New laptops pull the image. Later model, dataset, column, hyperparameter, and
rank-count changes require no build or push. New trainer code or architectures
and dependencies absent from the installed Transformers version still require
an image update. Remote Python scripts and quantisation-specific dependencies
are not enabled by this runner.

## Configurable ranks and architectures

`--world-size N` accepts any positive total rank count. Assign unique ranks from
`0` through `N-1`. Membership stays fixed during a run; restart when laptops join
or leave. Changing `N` does not require rebuilding.

Untied mT5 models use pipeline stages plus data-parallel replicas. The default
chooses the largest stage count dividing `N` that fits the model's computation
units. mT5-small has 18 units: shared embeddings, 8 encoder blocks, 8 decoder
blocks, and the output head. This limits stages per replica, not total ranks.
No stage is an idle relay. Override with `--pipeline-size P`, where `P` divides
`N` and fits the model. For example, `N=10,P=5` gives two five-stage replicas;
`N=30,P=10` gives three ten-stage replicas. Prime counts such as `N=29` default
to full-model replicas when there is no larger compatible stage count.

Other supported encoder–decoder and causal families, including T5/BART and
Llama/Qwen, use full-model data parallelism (`P=1`). Each laptop must fit the
whole model on that path. Causal training masks article/prompt tokens from the
loss and trains the summary suffix. It uses a text separator, not a chat template.

Each replica samples a different data shard. `--batch-size` is per replica, so
replica count changes the effective batch size. Gradients are normalized by
valid target tokens across replicas, synchronized, and clipped across unique
pipeline parameters. Distributed samplers can pad repeated examples to keep
replica lengths equal, including during validation.

The pipeline is synchronous, completing each micro-batch's forward/backward
before the next. Relative attention biases and encoder memory retain gradients
across stages. Transport is CPU-staged Gloo, including on CUDA. Additional
laptops do not guarantee faster training: communication and slow ranks matter.
Each host initially loads the full model on CPU before retaining its partition.

For data-replica gradient compression, see [GRADIENT_COMPRESSION.md](./GRADIENT_COMPRESSION.md).
Top-K retains Adafactor; DGC uses momentum SGD with local clipping and masking.
Both require at least two replicas and save their rank-local buffers in checkpoints.

## Run without source mounts

For Linux rank 0 with an NVIDIA GPU at `100.79.185.101`:

```bash
N=3 # Set to the total rank count for this run
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

Other Linux ranks change `--rank`. Windows setup and launch commands are in
[MT5_TAILSCALE.md](./MT5_TAILSCALE.md), including a GPU access check. CPU-only
hosts use `:cpu`, remove `--gpus all`, and set `--device cpu`. Device choices
can differ. All ranks must agree on experiment settings. The CUDA image supports
NVIDIA GPUs; it does not contain AMD/Intel GPU runtimes.

For a communication check, add `--smoke`, set `--steps 2`, and set
`--validation-batches 1` on every rank. Smoke uses a tiny random mT5 with
synthetic tokens, not the selected pretrained model/dataset.

## Switch models and datasets with flags

Replace these arguments on every rank, without rebuilding:

```text
# mT5, Tamil XL-Sum:
--model google/mt5-small --dataset csebuetnlp/xlsum --dataset-config tamil
--source-column text --target-column summary

# T5, CNN/DailyMail:
--model google-t5/t5-small --dataset abisee/cnn_dailymail --dataset-config 3.0.0
--source-column article --target-column highlights --source-prefix "summarize: "

# Qwen causal summarisation, Tamil XL-Sum:
--model Qwen/Qwen2.5-0.5B --dataset csebuetnlp/xlsum --dataset-config tamil
--source-column text --target-column summary
```

`--train-split`/`--validation-split` default to `train`/`validation` and accept HF
slice expressions. `--validation-batches 0` supports datasets without validation.
XL-Sum loads its original JSONL archive without executing the legacy script;
`--dataset-config` chooses the language, defaulting to Tamil for XL-Sum.
Other datasets use the installed Datasets loaders. Source/target columns must
contain nonempty strings. The task remains article-to-summary fine-tuning.

Local files work by mounting data (not code) into `/data` on every replica leader:

```text
-v /absolute/path/to/data:/data:ro
--dataset json --train-file /data/train.jsonl --validation-file /data/valid.jsonl
--source-column article --target-column summary
```

Use `--dataset csv` or `parquet` for those formats. Each replica leader loads and
tokenizes its data; all leaders need the same contents. Model IDs and local model
directories work; local models must exist at the same container path on all ranks.
Gated/private resources require suitable Hugging Face access credentials.

## Checkpoints and topology changes

Model/dataset downloads persist in `dtfm-hf-cache`. Each global rank saves config,
metrics, and `checkpoint.pt` under `/app/logs/mt5-tamil/rankN` in `dtfm-logs`.
Replica leaders save tokenizers. Saves happen at completion; periodic saves,
live worker membership changes, and automatic recovery are not implemented.

For the same topology, resume each rank's own checkpoint with `--resume-from`
and increase `--steps`. Optimizer state is restored, but shuffle/RNG traversal
restarts. Legacy fixed-stage optimizer checkpoints do not match new partitions.

To export weights, collect one directory per stage from a single replica:

```bash
sudo docker run --rm -v /absolute/path/to/checkpoints:/checkpoints \
  suryanarayanaant/dtfm:cpu python /app/scripts/assemble_mt5_checkpoint.py \
  --rank-dirs /checkpoints/rank0 /checkpoints/rank1 /checkpoints/rank2 \
  --output-dir /checkpoints/model
```

That list illustrates `P=3`; provide exactly `P` directories. A full-model data
replica requires only one directory. Other replicas hold synchronized copies.
When `N` or `P` changes, assemble/export the trained weights and select the merged
model with `--model`. Optimizer shards are not redistributed automatically.
Exported models load through `AutoModelForSeq2SeqLM` or `AutoModelForCausalLM`.

Sources: [mT5-small config](https://huggingface.co/google/mt5-small/raw/main/config.json),
[Qwen loading](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum).
