# Tamil summarisation with distributed mT5

This runner fine-tunes `google/mt5-small` on the Tamil configuration of
`csebuetnlp/xlsum`, using article `text` as input and `summary` as the target.
Training uses the official train split; validation uses the official validation
split. The test split remains held out.

## What runs on each machine

This initial implementation supports exactly **two pipeline ranks**:

- Rank 0: the entire encoder and the shared encoder/decoder input embedding.
- Rank 1: the entire decoder and its untied vocabulary output head.

Rank 0 sends encoder activations, decoder embeddings, source masks, and labels
to rank 1. Rank 1 sends gradients for both activations back to rank 0. Keeping
the shared input embedding on rank 0 preserves its combined encoder/decoder
gradient without synchronising a second large embedding table.

The pipeline is synchronous: one micro-batch completes forward and backward
before the next begins. Gradients accumulate over a batch, weighted by the
number of non-padding target tokens. Adafactor updates each stage; gradient
clipping uses the norm across both stages. Transport is CPU-staged Gloo,
including when either host computes on an NVIDIA GPU. The hosts can choose
different devices.

This is a separate encoder–decoder pipeline alongside `dist_runner.py`.
The existing GPT scheduler, dynamic layer migration, data parallel groups,
and CuPy transport do not apply to this runner. Arbitrary splits across three
or more machines are not implemented.

Both hosts initially load the full pretrained model on CPU and then keep only
their own stage. Allow RAM for that startup load. This is full fine-tuning;
GPU memory includes stage parameters, gradients, optimizer state, and
activations. Start with micro-batch size 1. The smoke test verifies transport,
not whether the full model fits. Models and datasets download at runtime and
are cached outside the image. Only rank 0 loads/tokenizes XL-Sum.

## Build once on each host

After obtaining the updated repository on each machine:

```bash
docker compose build train                  # CPU
# Or, on an NVIDIA host:
docker compose build gpu                   # CUDA
```

This rebuild adds Transformers and SentencePiece. Subsequent training settings
and model downloads do not require rebuilding. To edit this runner without
rebuilding, add `-v "$PWD:/app"` to the `docker compose run` commands after the
dependencies have been installed in the image.

For native Python, use Python 3.10 with the appropriate PyTorch installation:

```bash
python -m pip install -r requirements-mt5.txt
python scripts/train_mt5_pipeline.py --help
```

## Check communication first

Use two Linux machines on a network where rank 1 can reach rank 0.
Replace `192.168.1.10` with rank 0's reachable IPv4 address. Allow inbound
TCP 29500 and Gloo peer connections between these two machines. Choose the
correct local interface on **each host** (`ip -br address` lists interfaces).
Linux host networking lets Gloo peers connect to each other directly.
These multi-host examples assume Linux Docker Engine.

On machine 0:

```bash
docker compose -f compose.yaml -f compose.mt5-host.yaml run --rm --no-deps \
  -e GLOO_SOCKET_IFNAME=eth0 train mt5 \
  --rank 0 --dist-url tcp://192.168.1.10:29500 \
  --smoke --steps 2 --validation-batches 1
```

On machine 1, while machine 0 is waiting:

```bash
docker compose -f compose.yaml -f compose.mt5-host.yaml run --rm --no-deps \
  -e GLOO_SOCKET_IFNAME=eth0 train mt5 \
  --rank 1 --dist-url tcp://192.168.1.10:29500 \
  --smoke --steps 2 --validation-batches 1
```

The supplied `compose.mt5-host.yaml` override enables host networking for the
`train` and `gpu` services.

## Fine-tune on Tamil XL-Sum

After the smoke test succeeds, run on each machine with its own rank:

```bash
docker compose -f compose.yaml -f compose.mt5-host.yaml run --rm --no-deps \
  -e GLOO_SOCKET_IFNAME=eth0 train mt5 \
  --rank 0 --dist-url tcp://192.168.1.10:29500 \
  --batch-size 4 --micro-batch-size 1 --steps 1000 \
  --source-length 512 --target-length 128 \
  --validation-batches 10 --output-dir /app/logs/mt5-tamil
```

On machine 1, use the same command with `--rank 1`. Use `gpu` instead of
`train` on an NVIDIA host with NVIDIA Container Toolkit installed. All
training settings must match on both ranks; the compute devices may differ.
Do not use `rank0`/`rank1` Compose services: those run the original GPT smoke.

The XL-Sum archive is downloaded directly from the dataset's official Hugging
Face repository and loaded as JSONL, avoiding its legacy Python dataset script.
Default validation reports loss on the first 10 validation batches; use a
larger `--validation-batches` value to cover the full validation split.
Loss verifies training progress; assess generated Tamil summaries separately.

## Checkpoints and inference

Each host writes `metrics.jsonl`, `checkpoint.pt`, and `config.json` under
`logs/mt5-tamil/rank0` or `rank1`. Rank 0 also saves the tokenizer. A checkpoint
is saved after training and validation finish. There are no periodic saves.
Resume each host with `--resume-from /app/logs/mt5-tamil/rank0` (or `rank1`),
and set `--steps` to the new total step count. Both checkpoints must be from
the same run and step. Resume restores model and optimizer state, but starts
a new shuffled data traversal; it is not an exact replay of the data/RNG state.

Copy the two rank directories from the same run onto one machine, then merge:

```bash
docker compose run --rm train python scripts/assemble_mt5_checkpoint.py \
  --encoder-dir /app/logs/mt5-tamil/rank0 \
  --decoder-dir /app/logs/mt5-tamil/rank1 \
  --output-dir /app/logs/mt5-tamil/model
```

The merged directory is a normal Hugging Face model:

```python
from transformers import AutoTokenizer, MT5ForConditionalGeneration

path = "logs/mt5-tamil/model"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(path).eval()
inputs = tokenizer("உங்கள் தமிழ் செய்திக் கட்டுரையை இங்கே இடவும்.", return_tensors="pt",
                   truncation=True, max_length=512)
output = model.generate(**inputs, max_new_tokens=128, num_beams=4)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Sources: [mT5 model card](https://huggingface.co/google/mt5-small),
[XL-Sum dataset](https://huggingface.co/datasets/csebuetnlp/xlsum).
