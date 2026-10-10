# Gradient compression over Gloo/Tailscale

`scripts/train_mt5_pipeline.py` supports `--gradient-compression none|topk|dgc`.
The default remains dense synchronization with Adafactor. Docker's `summarize`
and `mt5` commands forward these flags; rebuild the image to include this code.

The BERT QA runner also supports `--gradient-compression none|topk`, retaining
AdamW. Its gradients are normalized by global feature count, rather than target
tokens. See [BERT-Large QA commands](./QA_BERT_LARGE_COMPRESSION.md) for a
four-rank example and QA-specific compression warm-up defaults.

## Choose the topology first

Compression exchanges parameter gradients between **data replicas of the same
pipeline stage**. It requires `world-size / pipeline-size >= 2`.
For four laptops, `--world-size 4 --pipeline-size 2` makes two two-stage replicas.
For two laptops, `--world-size 2 --pipeline-size 1` makes two full-model replicas,
and each host must fit the full model. Other supported model families already
use full-model replicas.

An mT5 run with one pipeline replica has no data-parallel gradient exchange to
compress. Selecting compression in that topology raises an actionable error.
Pipeline activations and their backward gradients still travel densely between
adjacent stages; this feature does not compress those messages.

## Keep Adafactor with Top-K error feedback

Add these arguments to the existing launch command on **every rank**, with a
topology containing at least two replicas:

```text
--gradient-compression topk
--compression-keep-ratio 0.01
--compression-warmup-steps 100
```

This selects the largest absolute accumulated gradients in bounded buckets,
retains unsent coordinates locally, gathers FP32 values and int32 bucket-local
indices, and sums overlapping coordinates. Gradients are already divided by the
global target-token count in this runner, so the reconstructed updates are
**summed**, without another division by replica count. The existing global
gradient clipping and Adafactor update follow synchronization. This is Top-K
with error feedback, rather than the complete momentum-SGD DGC algorithm.

Start with 1% kept and compare validation loss/summary quality against a dense
baseline before reducing it to `0.001` (0.1%). Unsent information is retained,
but its delay still changes training behavior.

## Run DGC with momentum SGD

Example for rank 0 of four hosts, with two stages per replica:

```bash
python scripts/train_mt5_pipeline.py \
  --rank 0 --world-size 4 --pipeline-size 2 \
  --dist-url tcp://100.79.185.101:29500 --device cuda \
  --model google/mt5-small --dataset csebuetnlp/xlsum --dataset-config tamil \
  --batch-size 4 --micro-batch-size 1 --steps 1000 \
  --optimizer sgd --momentum 0.9 --lr 0.0003 \
  --gradient-compression dgc --compression-keep-ratio 0.001 \
  --compression-warmup-steps 100 --output-dir logs/mt5-dgc
```

Launch ranks 1, 2, and 3 with the same settings and their own `--rank`.
Use your coordinator's Tailscale IP and the networking setup in
[MT5_TAILSCALE.md](./MT5_TAILSCALE.md). To check communication without model/data
downloads, add `--smoke --steps 2 --validation-batches 1` on every rank.
Docker users can append the same training flags after the image's `summarize`
command; Linux/Windows container networking is unchanged.

The DGC path performs these operations:

1. Clip each replica's gradients before accumulation, across its unique pipeline
   parameters, at `max-grad-norm / sqrt(replicas)`.
2. Update local momentum `u = momentum * u + g`, then residual `v += u`.
3. Transmit Top-K coordinates of `v` and reconstruct the sum from all data peers.
4. Clear both `v` and `u` at the **locally transmitted coordinates**, following
   the paper's momentum factor masking.
5. Apply plain SGD to the reconstructed velocity. Optimizer momentum is zero
   because the compressor already applied it.

`dgc` requires `--optimizer sgd`; combining DGC velocities with Adafactor is
rejected. Switching from Adafactor changes the optimizer, and the example LR
is only a starting value to tune against a dense SGD baseline. DGC uses local
clipping instead of the dense path's post-aggregation clipping.

Both modes start by keeping 25% and exponentially decrease to the target kept
fraction over `--compression-warmup-steps` (default 100). Set
`--compression-warmup-ratio` to change the starting fraction, and
`--compression-warmup-steps 0` to disable the schedule. The starting fraction
must be at least the target fraction. Learning rate remains constant; this
runner implements sparsity warm-up without a separate learning-rate schedule.

## Memory, payload, and measurement

Top-K runs on CPU in buckets of at most 4,194,304 elements. Change this with
`--compression-bucket-size`. This bounds temporary selection/receive memory and
combines small parameters, at the cost of two sparse collectives per bucket.
It selects within each bucket rather than doing one global Top-K. Each bucket
keeps at least one element. When values plus indices would occupy at least the
dense payload size, that bucket uses dense SUM exchange and flushes its buffers.

Persistent FP32 residual memory costs four bytes per parameter; DGC adds another
four bytes for momentum. For a 140M-parameter stage that is about 560 MB for
Top-K or 1.12 GB for DGC, excluding parameters, gradients, optimizer state, and
temporary buffers. Compression reduces network payload, not model memory.

For 140M FP32 gradients and `keep-ratio=0.001`, the sparse representation is
approximately `140M * 0.001 * (4 + 4) = 1.12 MB`, versus 560 MB dense (decimal
units). This is about 500x representation compression. Sparse all-gather
replicates each rank's payload to its peers, so collective traffic grows with
replica count. It also differs from dense all-reduce's traffic pattern. These
numbers exclude framing, control messages, pipeline traffic, and warm-up.
They do not predict total step time or guarantee unchanged model accuracy.

Per-step `metrics.jsonl` and console records include:

| Field | Meaning |
| --- | --- |
| `gradient_keep_ratio` | Scheduled kept fraction for that step |
| `gradient_selected_elements` | Local selected coordinates, including dense fallback buckets |
| `gradient_dense_bytes` | Dense representation size for this stage |
| `gradient_payload_bytes` | This rank's encoded payload size, including dense fallback |
| `gradient_payload_compression` | Dense bytes divided by payload bytes |
| `gradient_sync_seconds` | Local clipping for DGC, packing, selection, exchange, and reconstruction time |
| `seconds` | Full training-step time |

Payload fields describe the local representation, **not measured network bytes**.
Compare full step time and validation quality on the actual hosts. CUDA computation
still uses CPU-staged Gloo for communication.

## Resume and verification

Each rank's checkpoint saves its residuals, local momentum, completed compression
steps, configuration, and optimizer type. Resume each rank from its own directory
with the same compression, optimizer, clipping, and topology flags, increasing
`--steps`. Changing these settings on resume is rejected; export/assemble weights
and start a new run instead. Older dense Adafactor checkpoints remain supported.
Model export does not include pending local residuals as weight updates.
Dataset traversal and RNG restoration retain the runner's existing limitations.

```bash
python -m unittest discover -s tests -p 'test_gradient_compression.py' -v
python -m unittest discover -s tests -p 'test_mt5_pipeline.py' -v
```

The tests cover real Gloo subgroup exchange, overlapping/different selections,
residual conservation, missing gradients, momentum correction and masking,
dense fallback, warm-up state, and checkpoint resume versus uninterrupted tiny
pipeline training. They do not establish convergence on the real mT5 dataset.

References: [DGC paper, Lin et al., ICLR 2018](https://arxiv.org/abs/1712.01887),
[authors' PyTorch implementation](https://github.com/synxlin/deep-gradient-compression).
The paper reports 270x–600x compression without accuracy loss on its tested
tasks; those empirical results do not guarantee the same outcome for mT5
summarization over residential Tailscale links.
