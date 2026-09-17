# Pipeline scheduler

The scheduler connects network measurements and local compute profiles to real
GPipe stage assignments. It assigns a positive number of contiguous transformer
layers to every rank, subject to each device's supplied layer limit.

## Dynamic allocation from live resources

Use this mode to assign stages automatically from resources available when the
training processes start. No manual layer limits or saved compute profiles are
required:

```bash
python scripts/profile_to_launch_manifest.py verification.json \
  --hosts 100.64.0.10 100.64.0.11 100.64.0.12 \
  --devices cpu rocm xpu --dynamic --total-layers 12 --output launch.json
```

Run the emitted commands on their respective hosts. They carry
`DYNAMIC_TOTAL_LAYERS=12`. Model and batch settings come from the launcher's
usual `SEQ`, `EMBED`, `HEADS`, `BATCH`, `MICRO`, or `PROFILE=bert` settings.
Every rank must use matching settings. You can also pass
`--dynamic-total-layers 12` directly to `dist_runner.py`, with FP32, Gloo,
pipeline-only GPipe, `--use-offload false`, and `SeqClassification`.

The startup sequence is:

1. Initialize communication and load the endpoint datasets, then check that
   ranks agree on model, batch, scheduling policy, and vocabulary.
2. Check that available memory can accommodate the estimated minimum stages.
   Profile each rank's transformer computation and its endpoint component, if
   applicable. Calibration runs one rank at a time to bound profiling memory.
3. Read available memory again after calibration. Windows uses the smaller of
   available physical RAM and commit headroom. Linux uses `MemAvailable` bounded
   by visible cgroup limits. CUDA/ROCm and XPU also read global free GPU memory.
4. Give each rank one layer, then assign each remaining layer to the eligible
   rank with the smallest resulting compute time. A rank is eligible only while
   its estimated allocation fits all shared host and device memory budgets.
5. Check that every rank computed the same plan, record it, and construct the
   actual model stages using the negotiated counts.

There is no equal device weighting or equal division of RAM. Faster ranks can
receive more layers, and current memory pressure can reduce their assignments.
Similar resources can legitimately produce equal stage sizes. The network
manifest still determines rank order; this mode changes layer counts at startup.

Ranks on the same hostname share one host memory budget. Accelerator ranks also
share a device budget when they identify the same GPU UUID. If a backend does
not expose a UUID, same-backend adapters on that host conservatively share the
smallest observed device budget. Containers on one physical host should set the
same `SCHEDULER_HOST_ID`, or `--scheduler-host-id`, because container hostnames
can differ. Distinct physical hosts must have distinct identities.

By default each pool's usable budget is `0.7 * available_bytes - 256 MiB`.
`SCHEDULER_MEMORY_FRACTION` and `SCHEDULER_RESERVE_MB` adjust headroom, with
equivalent `--scheduler-memory-fraction` and `--scheduler-reserve-mb` flags.
These settings must agree across ranks. `SCHEDULER_WARMUP` and
`SCHEDULER_REPEATS` control calibration, defaulting to one warmup and three
timed samples. Timings reflect the compute environment at calibration time.

The memory model counts FP32 parameters and gradients exactly, estimates
checkpoint activations and attention workspace, and adds 25% padding. It also
includes embeddings, the classification head, and communication buffers. For
accelerators, the whole estimate is conservatively checked against host RAM as
well as GPU memory. These estimates are not measured peak-memory guarantees or
memory reservations. Other applications can change availability after discovery.
The allocator is a deterministic greedy heuristic, not an end-to-end throughput
optimizer.

Each rank writes `resource_schedule_rankN.json`. Metrics JSON embeds the agreed
plan, live resource readings, compute timings, memory budgets, estimated usage,
and layer ranges. MLflow receives actual assignment parameters after negotiation,
and its metrics artifact includes the plan. Version 3 launch manifests defer
stage allocation until startup; they do not embed a stale device snapshot.

Automatic memory discovery supports Windows/Linux CPU and runtime builds that
provide CUDA/ROCm or XPU free-memory queries. DirectML currently fails explicitly
in dynamic mode because its backend has no supported free-device-memory query.
Use the existing static path for DirectML. If the estimated resources cannot fit
all layers, or cannot fit one layer on each participating rank, all ranks fail
before model construction. This mode does not silently omit a rank or reduce
the model's total layer count.

Every fresh launch measures again. Optional periodic rebalancing can also change
the partition between completed optimizer steps, as described below.

### Rebalance during training

```bash
python scripts/profile_to_launch_manifest.py verification.json \
  --hosts 100.64.0.10 100.64.0.11 100.64.0.12 \
  --devices cpu rocm xpu --dynamic --total-layers 12 \
  --rebalance-every 20 --output launch.json
```

The emitted commands request a resource check after every 20 completed steps,
except after the final step. Direct launches use `--rebalance-every 20` with
`--dynamic-total-layers`, or `REBALANCE_EVERY=20` through `run_rank.sh`. The default
interval is zero, preserving startup-only behavior.

Each check profiles compute again, samples current memory, and proposes a new
layer allocation. All ranks pause at the same drained pipeline boundary, after
backward and the optimizer update. Profiling and migration preserve each rank's
random-number-generator state. The data iterator and completed-step count
continue from their current positions.

The proposed partition must reduce the estimated compute bottleneck by at least
10%, unless the current partition exceeds the newly estimated memory budgets.
Set `REBALANCE_MIN_IMPROVEMENT`, or `--rebalance-min-improvement`, to change this
threshold. The estimate does not include migration cost or guarantee a throughput
improvement. Choose an interval long enough to amortize calibration and transfer.

Migration uses stable global layer identities. Only layers whose owner changes
cross the network, as CPU-staged Gloo tensors. Embeddings and the output head
remain on the first and last ranks. Rank order, total layer count, communication
buffers, and model dimensions stay fixed. State transfer supports FP32 SGD with
one matching parameter group, including momentum buffers. Other optimizers,
mixed precision, and data parallelism are outside this path.

All ranks prepare storage before sends begin. Receivers check weights and SGD
state with SHA-256 checksums, construct candidate stages, and check converted
device state again. The old stage and optimizer remain intact until every rank
has prepared successfully. A preparation or capacity failure discards candidates
on all ranks and continues with the old partition. A unanimous vote swaps model
and optimizer references before the next step.

This transaction needs temporary headroom. The planner conservatively budgets
the replacement stage against currently free memory without crediting the old
stage's storage. A separate shared-pool check reserves room for incoming layers,
CPU transfer buffers, and optimizer state. A migration may therefore be skipped
even when the final partition would fit after deleting the old stage. There is
no fallback that discards trained weights to make room.

Transport failures or a crashed rank remain fatal. Rollback handles live-rank
preparation failures; it does not add persistent checkpoint recovery, rank
replacement, or membership changes.

Metrics contain `rebalance_events` with the completed step, decision, old/new
counts, elapsed time, moved-layer checksums, and committed resource plans.
`resource_schedule_rankN.json` retains the startup plan; final metrics contain
the final active plan and transition history. MLflow parameters describe the
initial allocation, while `assigned_layers`, `rebalance_seconds`, and
`rebalance_committed` track later decisions by step.

```bash
python scripts/verify_migration.py --world-size 3
python scripts/verify_launch_manifest.py --dynamic --world-size 3 \
  --total-layers 7 --rebalance-every 1
```

The migration verifier runs a fixed-partition baseline, two real state transfers,
an injected candidate-construction failure, and the resource controller with
controlled compute-rate changes. Every case trains for four steps. It checks
state checksums and RNG preservation across transitions, then compares all final
weights, nonempty SGD momentum state, and losses against the baseline. The live
launch verifier checks actual resource discovery and transition history;
similar resource readings may correctly result in no migration.

These tests prove local CPU/Gloo state continuity. Physical multi-host migration,
GPU migration, and throughput benefits still require separate validation.
PyTorch's [optimizer state format](https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.state_dict.html)
matches state by parameter order, so this implementation explicitly maps SGD
state by global layer and parameter name when ownership changes.

Memory API references: [Windows memory status](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex),
[CUDA memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html),
[XPU memory](https://docs.pytorch.org/docs/stable/generated/torch.xpu.memory.mem_get_info.html),
and [Linux cgroup limits](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).

### Verify dynamic launches locally

Run these serially to avoid competing Python/PyTorch processes exhausting the
machine's memory:

```bash
python scripts/verify_launch_manifest.py --dynamic --world-size 2 --total-layers 5
python scripts/verify_launch_manifest.py --dynamic --world-size 3 --total-layers 7
```

The verifier checks the full resource plan, agreement across ranks, assigned
layer ranges, completed training steps, and finite final-stage losses. Network
ordering still uses a test fixture; compute and available memory are measured
live on this machine. Resource-pressure tests use controlled snapshots to verify
that allocations change with free memory and compute rates.

## Static profile-based allocation

The earlier explicit-profile workflow remains available below.

### Profile each device

Run this on each participating machine with its training Python environment.
Use the same model and batch dimensions everywhere. Change `--measured-rank`,
`--device`, and `--max-layers` for that machine:

```bash
python scripts/profile_compute.py --measured-rank 0 --device cpu --max-layers 4 \
  --seq-length 32 --embedding-dim 64 --num-heads 4 \
  --batch-size 2 --micro-batch-size 1 --vocab-size 512 \
  --output compute-rank0.json
```

The profiler times a checkpointed transformer layer, embeddings, and the
two-class classification head separately. Each sample runs full-batch forward,
backward, and SGD work, retaining microbatch outputs before backward as GPipe
does. It records the median of five samples after two warmups. Accelerator work
is synchronized using the existing device backend. Profiling uses FP32 and
`SeqClassification`, matching the generated launch commands.

`max_layers` is a capacity supplied by the operator. Set a conservative limit
that leaves room for embeddings, the output head, optimizer state, communication
buffers, and retained microbatch activations. The profiler does not measure the
largest model that fits. CPU, CUDA/ROCm, XPU, and DirectML use the existing device
selection options. DirectML still requires its Python 3.11 environment.

For QQP, profile using the actual tokenizer vocabulary size. Generated commands
check it against the dataset on both endpoint ranks. The 512-token example is
for synthetic training. Timing profiles must be refreshed when the model,
batch, precision, device, or execution environment changes.

## Generate and run a schedule

Collect the compute files and the network `verification.json` produced by
`scripts/verify_comm_probe.py` on one machine:

```bash
python scripts/profile_to_launch_manifest.py verification.json \
  --hosts 100.64.0.10 100.64.0.11 100.64.0.12 \
  --devices cpu rocm directml \
  --compute-profiles compute-rank0.json compute-rank1.json compute-rank2.json \
  --total-layers 12 --output launch.json
```

Hosts and devices follow measured rank order. Compute files can be supplied in
any order; their rank identifiers must cover every measured rank exactly once,
their devices must match, and their model dimensions must agree.

Run each printed Bash command on its indicated host from the repository root.
The commands carry the complete `STAGE_LAYERS` vector and pin model/batch
dimensions from the compute profile. For synthetic training, export
`SYNTHETIC_DATA=true` first. For QQP, install the dataset as described in SETUP.md.
Set `PYTHON`, `LOG_DIR`, `ITERS`, and other training controls as usual. Do not
replace the emitted model or batch settings without profiling again.

Version 2 manifests include the original profiles, rank order, layer counts,
half-open layer ranges, and estimated stage compute times. For example, the
order `[0, 2, 1]` and counts `[1, 3, 2]` assign:

| Launch rank | Measured rank | Transformer layers |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 2 | 1, 2, 3 |
| 2 | 1 | 4, 5 |

Embeddings belong to the first stage and the classification head to the last.
Ranges describe this launch's model partition; they are not checkpoint keys or
a mechanism for moving existing weights between ranks.

## Selection and runtime checks

1. The existing network adapter finds the minimum bidirectional path through
   all ranks. Missing links remain unavailable. `--payload-bytes` controls the
   network ranking proxy, as documented in LAUNCH_MANIFEST.md.
2. For that fixed order, the scheduler minimizes the largest estimated stage
   compute time. A stage costs `layers * layer_ms` plus its embedding or head
   cost. It searches the finite set of possible stage costs and allocates every
   layer without exceeding device limits. Ties choose the lexicographically
   smallest layer-count vector.
3. The manifest validator recomputes the allocation and emitted commands.
   `dist_runner.py` resolves `--stage-layers` before model construction and
   MLflow tracking. All ranks compare assignments and training dimensions before
   transferring activations. Endpoint vocabulary checks also fail collectively.

The allocation is optimal for the supplied linear compute model and fixed
network order. The combined network/compute procedure is a two-phase heuristic;
estimated times do not predict end-to-end iteration time, communication overlap,
or pipeline fill/drain bubbles. It supports 2 through 12 ranks and up to 4096
transformer layers, with at least one layer per rank. The explicit-profile path
uses static pipeline-only GPipe. Use dynamic mode with a rebalance interval for
in-memory stage migration. Persistent checkpoint recovery, fault recovery, and
heterogeneous data parallelism remain unimplemented.

Omitting compute profiles preserves version 1 manifests and equal stage sizes.
Direct launches can supply `--stage-layers 1,3,2` on every rank. Metrics JSON
records local and total layer counts and layer ranges. Optional MLflow runs
record the same assignment parameters and retain the metrics artifact.

## Verification

```bash
python -m unittest discover -s tests
python scripts/verify_launch_manifest.py --scheduled --world-size 2
python scripts/verify_launch_manifest.py --scheduled --world-size 3
```

On Windows, use `.venv-cpu\Scripts\python.exe`. The launch verifier uses Git
Bash and writes fresh artifacts beneath `logs/smoke/`. It executes the emitted
commands and checks actual forward/backward/optimizer work, completed steps,
finite final-stage losses, and the assigned layer ranges. Scheduling fixtures
exercise unequal allocations and measured-rank remapping.

To exercise the full path with local CPU timing measurements, profile ranks 0
and 1 on this machine, then run:

```bash
python scripts/verify_launch_manifest.py --world-size 2 \
  --compute-profiles compute-rank0.json compute-rank1.json --total-layers 5
```

This proves local CPU execution with measured compute inputs and a synthetic
network fixture. Physical multi-host links, accelerator execution, memory limits,
model quality, and throughput improvements require separate hardware runs.
